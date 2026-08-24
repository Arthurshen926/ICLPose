"""Run pose-free full-token RADIO retrieval over physical parent/child surfaces."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Mapping, Sequence

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
from feature_extract.vfm.localization_goal_maplet.hierarchical_child_allocator import (
    SEMANTICS as HIERARCHICAL_CHILD_ALLOCATOR_SEMANTICS,
    allocate_parent_balanced_scene_children,
)
from feature_extract.vfm.localization_goal_maplet.layout_child_allocator_config import (
    SCORE_EVIDENCE_PER_AREA,
    SCORE_RAW_EVIDENCE,
    load_validate_layout_child_allocator_config,
    validate_layout_source_signature,
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
from feature_extract.vfm.localization_goal_maplet.scene_child_evidence import (
    SCENE_PARENT_MASK_SEMANTICS,
    aggregate_scene_child_evidence,
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
    parser.add_argument(
        "--layout_child_allocator_config",
        default="",
        help=(
            "signed seq10-frozen token-layout/parent-balanced child allocator; "
            "mutually exclusive with fine_support_area_fraction"
        ),
    )
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--include_trajectories", nargs="+", default=[])
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument(
        "--allow_legacy_coordinate_misaligned_control",
        action="store_true",
        help="diagnostic-only opt-in for old canonical fields without raw-RADIO coordinate alignment",
    )
    parser.add_argument(
        "--allow_unpromoted_mapper_control",
        action="store_true",
        help=(
            "explicitly consume an unpromoted mapper/canonical/calibration "
            "chain as a diagnostic control; outputs remain non-promotable"
        ),
    )
    parser.add_argument(
        "--allow_query_route_overlap_control",
        action="store_true",
        help="explicitly allow query-route overlap as a control-only retrieval run",
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


def _validate_promotion_contract(
    field: CanonicalSurfaceField,
    calibration: ValidityCalibration | None,
    *,
    allow_unpromoted_mapper_control: bool,
) -> tuple[bool, list[str]]:
    field_eligible = bool(field.metadata.get("promotion_eligible", False))
    calibration_eligible = bool(
        calibration is not None
        and calibration.metadata.get("promotion_eligible", False)
    )
    blockers: list[str] = []
    if not field_eligible:
        blockers.append("canonical_field_not_promotion_eligible")
    if calibration is not None and not calibration_eligible:
        blockers.append("validity_calibration_not_promotion_eligible")
    eligible = bool(field_eligible and (calibration is None or calibration_eligible))
    if not eligible and not bool(allow_unpromoted_mapper_control):
        raise ValueError(
            "pure retrieval refuses an unpromoted canonical/calibration chain; "
            "pass --allow_unpromoted_mapper_control for a diagnostic control"
        )
    return eligible, blockers


def _query_split_audit(
    query_routes: set[str],
    field_metadata: Mapping[str, object],
    mapper_metadata: Mapping[str, object],
    calibration_metadata: Mapping[str, object],
) -> dict[str, object]:
    mapping_routes = {
        str(value) for value in field_metadata.get("mapping_trajectory_ids", ())
    }
    field_excluded = {
        str(value) for value in field_metadata.get("excluded_trajectory_ids", ())
    }
    mapper_fit = {
        str(value) for value in mapper_metadata.get("training_trajectory_ids", ())
    }
    mapper_validation = {
        str(value) for value in mapper_metadata.get("validation_trajectory_ids", ())
    }
    mapper_holdout = {
        str(value)
        for value in mapper_metadata.get("strict_holdout_trajectory_ids", ())
    }
    calibration_fit = {
        str(value) for value in calibration_metadata.get("fit_trajectory_ids", ())
    }
    blockers: list[str] = []
    if not query_routes:
        blockers.append("query_routes_not_explicit")
    if query_routes & mapping_routes:
        blockers.append("query_route_present_in_canonical_fusion")
    if not query_routes.issubset(field_excluded):
        blockers.append("canonical_field_does_not_declare_query_route_exclusion")
    if query_routes & (mapper_fit | mapper_validation):
        blockers.append("query_route_used_for_mapper_fit_or_selection")
    if not query_routes.issubset(mapper_holdout):
        blockers.append("mapper_does_not_declare_query_route_holdout")
    if query_routes & calibration_fit:
        blockers.append("query_route_used_for_validity_calibration")
    return {
        "disjoint": not blockers,
        "query_trajectory_ids": sorted(query_routes),
        "canonical_mapping_trajectory_ids": sorted(mapping_routes),
        "canonical_excluded_trajectory_ids": sorted(field_excluded),
        "mapper_training_trajectory_ids": sorted(mapper_fit),
        "mapper_validation_trajectory_ids": sorted(mapper_validation),
        "mapper_strict_holdout_trajectory_ids": sorted(mapper_holdout),
        "validity_calibration_fit_trajectory_ids": sorted(calibration_fit),
        "blockers": blockers,
    }


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
    if str(contract.metadata.get("canonical_field_file_sha256", "")) != compute_file_sha256(
        field_path
    ):
        raise ValueError("field feature contract does not bind canonical field bytes")
    calibration = ValidityCalibration.load_json(calibration_path)
    promotion_eligible, promotion_blockers = _validate_promotion_contract(
        field,
        calibration,
        allow_unpromoted_mapper_control=bool(
            args.allow_unpromoted_mapper_control
        ),
    )
    for name, expected in (
        ("physical_map_sha256", physical.content_sha256),
        ("canonical_field_sha256", field.content_sha256),
        ("canonical_field_file_sha256", compute_file_sha256(field_path)),
        ("surface_mapper_file_sha256", compute_file_sha256(mapper_path)),
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
    layout_config_path = (
        Path(args.layout_child_allocator_config).resolve()
        if str(args.layout_child_allocator_config)
        else None
    )
    fine_child_area = (
        child_surface_area_m2(physical)
        if float(args.fine_support_area_fraction) > 0.0
        or layout_config_path is not None
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
    selected_query_routes = {
        str(record.image_id).split("/", 1)[0] for record in records
    }
    layout_config: dict[str, object] | None = None
    layout_config_file_sha256: str | None = None
    if layout_config_path is not None:
        if float(args.fine_support_area_fraction) > 0.0:
            raise ValueError(
                "layout child allocator and fine-support area selection are mutually exclusive"
            )
        layout_config, tuning_signature = (
            load_validate_layout_child_allocator_config(
                layout_config_path,
                physical,
                physical_path=physical_path,
                query_routes=selected_query_routes,
            )
        )
        if (
            int(layout_config["maximum_children"])
            != int(args.maximum_scene_children)
            or not np.isclose(
                float(layout_config["maximum_primitive_iou"]),
                float(args.maximum_child_primitive_iou),
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise ValueError("layout config and retrieval child budget differ")
        deployment_signature = {
            "physical_map_file_sha256": compute_file_sha256(physical_path),
            "canonical_field_file_sha256": compute_file_sha256(field_path),
            "surface_mapper_file_sha256": compute_file_sha256(mapper_path),
            "field_feature_contract_file_sha256": compute_file_sha256(contract_path),
            "validity_calibration_file_sha256": compute_file_sha256(calibration_path),
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
            "maximum_parent_candidates": int(args.maximum_parent_candidates),
            "maximum_child_candidates": int(args.maximum_child_candidates),
            "maximum_scene_parents": int(args.maximum_scene_parents),
            "child_probability_semantics": CHILD_PROBABILITY_SEMANTICS,
            "token_height": 36,
            "token_width": 64,
            "pool_sizes": list(POOL_SIZES),
            "pool_weights": list(POOL_WEIGHTS),
        }
        validate_layout_source_signature(tuning_signature, deployment_signature)
        layout_config_file_sha256 = compute_file_sha256(layout_config_path)
    query_split_audit = _query_split_audit(
        selected_query_routes, field.metadata, mapper_metadata, calibration.metadata
    )
    if not bool(query_split_audit["disjoint"]):
        if not bool(args.allow_query_route_overlap_control):
            raise ValueError(
                "pure retrieval query routes overlap fitting/calibration: "
                + ",".join(str(value) for value in query_split_audit["blockers"])
            )
        promotion_eligible = False
        promotion_blockers.extend(
            str(value) for value in query_split_audit["blockers"]
        )
    if layout_config is not None:
        query_split_audit.update({
            "allocator_tuning_trajectory_ids": [
                str(layout_config["tuning_route"])
            ],
            "allocator_tuning_query_disjoint": True,
        })
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
            if layout_config is not None and str(
                result.metadata.get(
                    "layout_child_allocator_config_content_sha256", ""
                )
            ) != str(layout_config["content_sha256"]):
                raise ValueError("existing retrieval uses a different layout config")
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
        layout_evidence_audit: dict[str, object] | None = None
        if layout_config is None:
            child_scene_score = aggregate_sparse_token_evidence(
                token_xy,
                child.candidate_child_rows,
                child.candidate_probabilities,
                entity_count=physical.child_parent_rows.size,
                token_height=36,
                token_width=64,
            )
        else:
            child_scene_score, layout_evidence_audit = (
                aggregate_scene_child_evidence(
                    token_xy,
                    child.candidate_child_rows,
                    child.candidate_probabilities,
                    physical,
                    token_height=36,
                    token_width=64,
                    semantics=str(layout_config["child_evidence_semantics"]),
                    local_block_size=int(
                        layout_config["child_evidence_local_block_size"]
                    ),
                    scene_parent_ids=physical.maplet_ids[parent_order],
                )
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
        elif layout_config is not None:
            fine_selection = None
            assert fine_child_area is not None
            if str(layout_config["child_score_policy"]) == SCORE_RAW_EVIDENCE:
                allocation_score = child_scene_score
            elif str(layout_config["child_score_policy"]) == SCORE_EVIDENCE_PER_AREA:
                allocation_score = np.divide(
                    child_scene_score,
                    fine_child_area,
                    out=np.zeros_like(child_scene_score),
                    where=fine_child_area > 0.0,
                )
            else:
                raise ValueError("unknown frozen layout child score policy")
            allocation = allocate_parent_balanced_scene_children(
                physical.maplet_ids[parent_order],
                ranked_parent_score,
                allocation_score,
                physical,
                parent_mass_fraction=float(
                    layout_config["parent_mass_fraction"]
                ),
                maximum_children=int(layout_config["maximum_children"]),
                maximum_primitive_iou=float(
                    layout_config["maximum_primitive_iou"]
                ),
            )
            scene_child_rows = allocation.child_rows
            scene_child_scores = allocation.child_scores
            suppressed = allocation.suppressed_duplicate_count
            connected_supports = connected_fine_support_components(
                scene_child_rows,
                physical,
                maximum_normal_angle_degrees=float(
                    layout_config["maximum_normal_angle_degrees"]
                ),
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
                    else (
                        HIERARCHICAL_CHILD_ALLOCATOR_SEMANTICS
                        if layout_config is not None
                        else "fixed_topk_block_peak_score_v1"
                    )
                ),
                "scene_child_evidence_semantics": (
                    str(layout_config["child_evidence_semantics"])
                    if layout_config is not None
                    else SCENE_AGGREGATION
                ),
                "scene_child_score_policy": (
                    str(layout_config["child_score_policy"])
                    if layout_config is not None
                    else "raw_child_evidence_v1"
                ),
                "scene_child_candidate_parent_mask_semantics": (
                    SCENE_PARENT_MASK_SEMANTICS
                    if layout_config is not None
                    else "none"
                ),
                "layout_child_allocator_config": (
                    str(layout_config_path)
                    if layout_config_path is not None
                    else None
                ),
                "layout_child_allocator_config_file_sha256": (
                    layout_config_file_sha256
                    if layout_config is not None
                    else None
                ),
                "layout_child_allocator_config_content_sha256": (
                    str(layout_config["content_sha256"])
                    if layout_config is not None
                    else None
                ),
                "layout_child_allocator_tuning_route": (
                    str(layout_config["tuning_route"])
                    if layout_config is not None
                    else None
                ),
                "layout_child_retained_evidence_fraction_after_parent_mask": (
                    float(
                        layout_evidence_audit[
                            "retained_child_evidence_fraction_after_parent_mask"
                        ]
                    )
                    if layout_evidence_audit is not None
                    else None
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
                "promotion_eligible": promotion_eligible,
                "promotion_blockers": promotion_blockers,
                "control_only": bool(not promotion_eligible),
                "query_split_audit": query_split_audit,
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
        "promotion_eligible": promotion_eligible,
        "promotion_blockers": promotion_blockers,
        "control_only": bool(not promotion_eligible),
        "query_split_audit": query_split_audit,
        "field_feature_contract_sha256": contract.content_sha256,
        "validity_calibration_sha256": calibration.content_sha256,
        "method": "full_radio_36x64_parent_then_child_physical_retrieval",
        "child_probability_semantics": CHILD_PROBABILITY_SEMANTICS,
        "child_probability_is_calibrated_credible_mass": False,
        "scene_aggregation": SCENE_AGGREGATION,
        "scene_child_evidence_semantics": (
            str(layout_config["child_evidence_semantics"])
            if layout_config is not None
            else SCENE_AGGREGATION
        ),
        "scene_child_score_policy": (
            str(layout_config["child_score_policy"])
            if layout_config is not None
            else "raw_child_evidence_v1"
        ),
        "scene_child_candidate_parent_mask_semantics": (
            SCENE_PARENT_MASK_SEMANTICS
            if layout_config is not None
            else "none"
        ),
        "layout_child_allocator_config": (
            str(layout_config_path) if layout_config_path is not None else None
        ),
        "layout_child_allocator_config_file_sha256": (
            layout_config_file_sha256 if layout_config is not None else None
        ),
        "layout_child_allocator_config_content_sha256": (
            str(layout_config["content_sha256"])
            if layout_config is not None
            else None
        ),
        "layout_child_allocator_tuning_route": (
            str(layout_config["tuning_route"])
            if layout_config is not None
            else None
        ),
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
