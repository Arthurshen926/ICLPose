"""Oracle audit for pose-conditioned Goal-Maplet query-edge geometry.

This G20.4-A diagnostic does not train or rerank the production candidate
pool.  It compares a runtime-supported truth-parent configuration against the
current selected and near-phase runtime configurations on the three frozen B1
failures.  Ground truth is used only to construct/report oracle controls and
to identify near-phase diagnostic candidates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_pose_modes import _camera
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
from feature_extract.vfm.localization_goal_maplet.feature_contract import (
    FieldFeatureContract,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.pfir import (
    ContributorLabels,
    contributor_multiscale_maplet_distribution,
)
from feature_extract.vfm.localization_goal_maplet.physical_instance_readout import (
    encode_physical_instance_regions,
    load_physical_instance_readout,
)
from feature_extract.vfm.localization_goal_maplet.physical_map import (
    GoalMapletPhysicalMap,
)
from feature_extract.vfm.localization_goal_maplet.pose_proposal import _score_pose
from feature_extract.vfm.localization_goal_maplet.query_edge_factor import (
    QueryEdgeGraph,
    build_query_edge_graph,
    score_pose_conditioned_query_edges,
)
from feature_extract.vfm.localization_goal_maplet.query_support import (
    aggregate_group_descriptors,
    aggregate_group_posteriors,
    all_token_coordinates,
    group_tokens_after_retrieval,
)
from feature_extract.vfm.localization_goal_maplet.retrieval import (
    ValidityCalibration,
    retrieve_maplet_posterior,
)
from feature_extract.vfm.query_to_3d_matching import (
    camera_matrix_and_distortion,
    pnp_pose_error,
)


MODE = "actual_parent_actual_child"
ORACLE_MODE = "oracle_parent_actual_child"


def _trajectory(image_id: str) -> str:
    return str(image_id).replace("\\", "/").split("/", 1)[0]


def _rows_by_image(report: dict[str, object]) -> dict[str, dict[str, object]]:
    return {str(row["image_id"]): row for row in report.get("rows", ())}


def _pose(detail: dict[str, object]) -> np.ndarray:
    value = np.asarray(detail["pose_w2c"], dtype=np.float64)
    if value.shape != (4, 4) or not np.all(np.isfinite(value)):
        raise ValueError("candidate detail contains an invalid pose")
    return value


def _solve_configuration_pose(
    xy_px: np.ndarray,
    child_rows: np.ndarray,
    support_mask: np.ndarray,
    physical: GoalMapletPhysicalMap,
    camera,
    gt_pose: np.ndarray,
    *,
    reprojection_px: float = 32.0,
) -> tuple[np.ndarray | None, dict[str, object]]:
    selected = np.flatnonzero(
        np.asarray(support_mask, dtype=bool) & (np.asarray(child_rows) >= 0)
    )
    xyz = physical.child_centers[np.asarray(child_rows, dtype=np.int64)[selected]]
    unique = int(np.unique(np.round(xyz, 5), axis=0).shape[0]) if xyz.size else 0
    report: dict[str, object] = {
        "correspondence_count": int(selected.size),
        "unique_child_center_count": unique,
        "inlier_count": 0,
        "translation_m": None,
        "rotation_deg": None,
    }
    if selected.size < 6 or unique < 6:
        return None, report
    matrix, distortion = camera_matrix_and_distortion(camera)
    cv2.setRNGSeed(194917)
    try:
        success, rotation, translation, inliers = cv2.solvePnPRansac(
            xyz.astype(np.float64),
            np.asarray(xy_px, dtype=np.float64)[selected],
            matrix,
            distortion,
            iterationsCount=6000,
            reprojectionError=float(reprojection_px),
            confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP,
        )
    except cv2.error:
        success, inliers = False, None
    if not success or inliers is None or len(inliers) < 6:
        return None, report
    inlier = np.asarray(inliers, dtype=np.int64).reshape(-1)
    try:
        rotation, translation = cv2.solvePnPRefineLM(
            xyz[inlier].astype(np.float64),
            np.asarray(xy_px, dtype=np.float64)[selected[inlier]],
            matrix,
            distortion,
            rotation,
            translation,
        )
    except cv2.error:
        pass
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = cv2.Rodrigues(rotation)[0]
    pose[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    error = pnp_pose_error(pose, gt_pose)
    report.update({
        "inlier_count": int(inlier.size),
        "translation_m": float(error.translation_m),
        "rotation_deg": float(error.rotation_deg),
    })
    return pose, report


def _remap_graph(graph: QueryEdgeGraph, selected: np.ndarray) -> QueryEdgeGraph:
    rows = np.asarray(selected, dtype=np.int64).reshape(-1)
    left, right = rows[graph.source], rows[graph.target]
    return QueryEdgeGraph(
        np.minimum(left, right), np.maximum(left, right), graph.kind,
        graph.high_displacement, graph.distinctive_context,
    )


def _truth_supported_parent_posterior(
    truth: np.ndarray,
    truth_null: np.ndarray,
    parent_ids: np.ndarray,
    parent_probability: np.ndarray,
    physical: GoalMapletPhysicalMap,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    truth_row = np.argmax(np.asarray(truth), axis=1)
    truth_id = physical.maplet_ids[truth_row]
    output_id = np.full((truth_row.size, 1), -1, dtype=np.int64)
    output_probability = np.zeros((truth_row.size, 1), dtype=np.float32)
    supported = np.zeros((truth_row.size,), dtype=bool)
    for support in range(truth_row.size):
        if float(1.0 - truth_null[support]) <= 1e-6:
            continue
        slot = np.flatnonzero(
            (parent_ids[support] == int(truth_id[support]))
            & (parent_probability[support] > 0.0)
        )
        if slot.size:
            supported[support] = True
            output_id[support, 0] = int(truth_id[support])
            output_probability[support, 0] = float(
                parent_probability[support, int(slot[0])]
            )
    output_null = np.where(
        supported,
        np.maximum(np.asarray(truth_null, dtype=np.float64), 1.0 - output_probability[:, 0]),
        1.0,
    ).astype(np.float32)
    return output_id, output_probability, output_null, supported


def _best_children(posterior, supports: np.ndarray) -> np.ndarray:
    output = np.full((posterior.candidate_child_rows.shape[0],), -1, dtype=np.int64)
    for support in np.flatnonzero(np.asarray(supports, dtype=bool)).tolist():
        rows = posterior.candidate_child_rows[support]
        probability = posterior.candidate_probabilities[support]
        valid = (rows >= 0) & (probability > 0.0)
        if np.any(valid):
            options = np.flatnonzero(valid)
            slot = int(options[np.argmax(probability[options])])
            output[support] = int(rows[slot])
    return output


def _configuration_score(
    graph: QueryEdgeGraph,
    grouped,
    child_rows: np.ndarray,
    pose: np.ndarray,
    physical: GoalMapletPhysicalMap,
    camera,
    support_mask: np.ndarray,
) -> dict[str, dict[str, float | int | None]]:
    return score_pose_conditioned_query_edges(
        graph,
        grouped.xy,
        grouped.extent,
        child_rows,
        pose,
        physical,
        camera,
        support_mask=support_mask,
    )


def _paired_margin(
    truth_score: dict[str, dict[str, float | int | None]],
    wrong_score: dict[str, dict[str, float | int | None]],
) -> dict[str, dict[str, float | int | None]]:
    output = {}
    for name in truth_score:
        truth = truth_score[name]
        wrong = wrong_score[name]
        left, right = truth["vector_score"], wrong["vector_score"]
        output[name] = {
            "edge_count": int(truth["edge_count"]),
            "truth_vector_score": left,
            "wrong_vector_score": right,
            "truth_minus_wrong_margin": (
                float(left) - float(right) if left is not None and right is not None else None
            ),
            "truth_order_consistency": truth["order_consistency"],
            "wrong_order_consistency": wrong["order_consistency"],
            "truth_relative_scale_score": truth["relative_scale_score"],
            "wrong_relative_scale_score": wrong["relative_scale_score"],
        }
    return output


def _validate_lineage(
    args,
    physical: GoalMapletPhysicalMap,
    field: CanonicalSurfaceField,
    reports: dict[str, dict[str, object]],
) -> dict[str, object]:
    for name, report in reports.items():
        if report.get("physical_map_sha256") != physical.content_sha256:
            raise ValueError(f"{name} pool and physical map differ")
        if report.get("canonical_field_sha256") != field.content_sha256:
            raise ValueError(f"{name} pool and canonical field differ")
    if not bool(reports["conditioned"].get("parent_conditioned_child_enumeration")):
        raise ValueError("conditioned pool predates parent-conditioned child enumeration")
    if not (
        float(reports["pre_nms"].get("translation_nms_m", -1.0)) == 0.0
        and float(reports["pre_nms"].get("rotation_nms_deg", -1.0)) == 0.0
    ):
        raise ValueError("diagnostic near-phase pool is not pre-NMS")
    crossfit = json.loads(Path(args.crossfit_evaluation).read_text())
    if (
        crossfit.get("stage")
        != "goal_maplet_phase_operator_outer_feature_crossfit_evaluation"
        or not bool(crossfit.get("feature_pipeline_outer_crossfit"))
    ):
        raise ValueError("query-edge audit lacks outer feature cross-fit lineage")
    held = {_trajectory(image_id) for image_id in args.image_ids}
    if held != {str(value) for value in crossfit.get("heldout_trajectories", ())}:
        raise ValueError("query-edge audit images and held fold differ")
    contract = dict(crossfit.get("candidate_generator_contract") or {})
    if contract.get("physical_map_sha256") != physical.content_sha256:
        raise ValueError("cross-fit evaluation and physical map differ")
    if contract.get("canonical_field_sha256") != field.content_sha256:
        raise ValueError("cross-fit evaluation and canonical field differ")
    return crossfit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument("--surface_mapper", required=True)
    parser.add_argument("--field_feature_contract", required=True)
    parser.add_argument("--validity_calibration", required=True)
    parser.add_argument("--physical_instance_readout", required=True)
    parser.add_argument("--crossfit_evaluation", required=True)
    parser.add_argument("--conditioned_candidate_pool", required=True)
    parser.add_argument("--pre_nms_candidate_pool", required=True)
    parser.add_argument("--oracle_candidate_pool", required=True)
    parser.add_argument("--image_ids", nargs="+", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--parent_candidates", type=int, default=64)
    parser.add_argument("--child_candidates", type=int, default=64)
    parser.add_argument("--maximum_edge_supports", type=int, default=96)
    parser.add_argument("--near_phase_candidates", type=int, default=8)
    parser.add_argument("--grouping_cosine", type=float, default=0.96)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite query-edge oracle audit")
    physical = GoalMapletPhysicalMap.load_npz(Path(args.physical_map))
    field = CanonicalSurfaceField.load_npz(Path(args.canonical_field))
    contract = FieldFeatureContract.load_json(Path(args.field_feature_contract))
    contract.validate(field, query_readout_path=Path(args.surface_mapper))
    mapper, _ = load_surface_maplet_mapper(
        Path(args.surface_mapper), device=str(args.device),
    )
    instance, instance_metadata = load_physical_instance_readout(
        Path(args.physical_instance_readout), device=str(args.device),
    )
    if instance_metadata.get("physical_map_sha256") != physical.content_sha256:
        raise ValueError("physical readout and map differ")
    if instance_metadata.get("canonical_field_sha256") != field.content_sha256:
        raise ValueError("physical readout and field differ")
    readout = readout_canonical_field(field, physical)
    parent_descriptor = instance.project_numpy(
        readout.parent_descriptors, role="context", device=str(args.device),
    )
    child_descriptor = instance.project_numpy(
        readout.child_descriptors, role="local", device=str(args.device),
    )
    calibration = ValidityCalibration.load_json(Path(args.validity_calibration))
    pool_paths = {
        "conditioned": Path(args.conditioned_candidate_pool),
        "pre_nms": Path(args.pre_nms_candidate_pool),
        "oracle": Path(args.oracle_candidate_pool),
    }
    reports = {name: json.loads(path.read_text()) for name, path in pool_paths.items()}
    crossfit = _validate_lineage(args, physical, field, reports)
    report_rows = {name: _rows_by_image(report) for name, report in reports.items()}
    requested = {str(value) for value in args.image_ids}
    contributor_paths: dict[str, Path] = {}
    for path in sorted(Path(args.contributors).glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        image_id = str(metadata["image_id"])
        if image_id in requested:
            contributor_paths[image_id] = path
    missing = requested - set(contributor_paths)
    if missing:
        raise ValueError(f"missing requested contributor labels: {sorted(missing)}")

    rows = []
    for image_id in args.image_ids:
        path = contributor_paths[str(image_id)]
        labels = ContributorLabels.load_npz(path)
        camera = _camera(path)
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        with np.load(Path(str(metadata["token_path"])), allow_pickle=False) as data:
            raw = np.asarray(data["radio_final"], dtype=np.float32)
        mapped = mapper.project(raw).measurement_context
        _, token_xy = all_token_coordinates(int(raw.shape[1]), int(raw.shape[2]))
        context_descriptor = encode_physical_instance_regions(
            instance, mapped, token_xy, role="context", device=str(args.device),
        )
        local_descriptor = encode_physical_instance_regions(
            instance, mapped, token_xy, role="local", device=str(args.device),
        )
        parent_ids, parent_probability, parent_null, _ = retrieve_maplet_posterior(
            context_descriptor,
            parent_descriptor,
            physical.maplet_ids,
            readout.parent_coverage > 0.0,
            maximum_candidates=int(args.parent_candidates),
            temperature=0.07,
            null_similarity_center=float(calibration.center),
            null_similarity_scale=float(calibration.scale),
        )
        grouped = group_tokens_after_retrieval(
            token_xy,
            context_descriptor,
            parent_ids[:, 0],
            token_height=int(raw.shape[1]),
            token_width=int(raw.shape[2]),
            image_width=int(camera.width),
            image_height=int(camera.height),
            descriptor_half_size_tokens=2.0,
            minimum_descriptor_cosine=float(args.grouping_cosine),
        )
        grouped_parent_ids, grouped_parent_probability, grouped_parent_null = (
            aggregate_group_posteriors(
                parent_ids,
                parent_probability,
                parent_null,
                grouped.member_offsets,
                grouped.member_token_indices,
                maximum_candidates=int(args.parent_candidates),
            )
        )
        grouped_local = aggregate_group_descriptors(
            local_descriptor,
            grouped.member_offsets,
            grouped.member_token_indices,
        )
        actual_child = retrieve_children_given_parents(
            grouped_local,
            grouped_parent_ids,
            grouped_parent_probability,
            grouped_parent_null,
            child_descriptor,
            readout.child_coverage,
            physical,
            maximum_child_candidates=int(args.child_candidates),
            temperature=0.07,
        )
        truth_parent, truth_parent_null = contributor_multiscale_maplet_distribution(
            labels,
            physical,
            token_xy,
            token_height=int(raw.shape[1]),
            token_width=int(raw.shape[2]),
            pool_sizes=(1,),
            pool_weights=(1.0,),
            group_member_offsets=grouped.member_offsets,
            group_member_token_indices=grouped.member_token_indices,
        )
        truth_parent_rows = np.argmax(truth_parent, axis=1)
        oracle_parent_ids, oracle_parent_probability, oracle_parent_null, parent_supported = (
            _truth_supported_parent_posterior(
                truth_parent,
                truth_parent_null,
                grouped_parent_ids,
                grouped_parent_probability,
                physical,
            )
        )
        oracle_child = retrieve_children_given_parents(
            grouped_local,
            oracle_parent_ids,
            oracle_parent_probability,
            oracle_parent_null,
            child_descriptor,
            readout.child_coverage,
            physical,
            maximum_child_candidates=int(args.child_candidates),
            temperature=0.07,
        )
        query_confidence = 1.0 - grouped_parent_null
        edge_supports = np.argsort(-query_confidence, kind="stable")[:
            int(args.maximum_edge_supports)
        ]
        edge_support_mask = np.zeros((grouped.xy.shape[0],), dtype=bool)
        edge_support_mask[edge_supports] = True
        local_graph = build_query_edge_graph(
            grouped.xy[edge_supports], grouped.descriptors[edge_supports],
        )
        graph = _remap_graph(local_graph, edge_supports)
        xy_px = grouped.xy * np.asarray([camera.width, camera.height], dtype=np.float64)
        extent_px = grouped.extent * np.asarray([camera.width, camera.height], dtype=np.float64)
        truth_children = _best_children(
            oracle_child, edge_support_mask & parent_supported,
        )
        naive_truth_pose, naive_truth_pose_report = _solve_configuration_pose(
            xy_px,
            truth_children,
            edge_support_mask & parent_supported,
            physical,
            camera,
            labels.pose_w2c,
        )
        oracle_details = list(
            report_rows["oracle"][image_id]["mode_details"][ORACLE_MODE]
        )
        if not oracle_details:
            raise RuntimeError(f"oracle-parent candidate pool is empty for {image_id}")
        oracle_detail = oracle_details[0]
        truth_pose = _pose(oracle_detail)
        truth_pose_report = {
            "source": "frozen_oracle_parent_actual_child_candidate_pool",
            "translation_m": float(oracle_detail["translation_m"]),
            "rotation_deg": float(oracle_detail["rotation_deg"]),
            "supporting_region_count": int(oracle_detail["supporting_region_count"]),
        }
        truth_primary = _configuration_score(
            graph,
            grouped,
            truth_children,
            truth_pose,
            physical,
            camera,
            edge_support_mask,
        )
        truth_gt_control = _configuration_score(
            graph,
            grouped,
            truth_children,
            labels.pose_w2c,
            physical,
            camera,
            edge_support_mask,
        )

        selected_details = list(
            report_rows["conditioned"][image_id]["mode_details"][MODE]
        )
        if not selected_details:
            raise RuntimeError(f"current candidate pool is empty for {image_id}")
        current_detail = selected_details[0]
        current_pose = _pose(current_detail)
        _, _, current_children = _score_pose(
            current_pose, xy_px, extent_px, actual_child, physical, camera,
        )
        common = edge_support_mask & (truth_children >= 0) & (current_children >= 0)
        current_common_count = int(np.sum(common))
        truth_current_common = _configuration_score(
            graph, grouped, truth_children, truth_pose, physical, camera, common,
        )
        current_score = _configuration_score(
            graph, grouped, current_children, current_pose, physical, camera, common,
        )
        current_margin = _paired_margin(truth_current_common, current_score)
        current_parent_rows = np.full(current_children.shape, -1, dtype=np.int64)
        current_assigned = current_children >= 0
        current_parent_rows[current_assigned] = physical.child_parent_rows[
            current_children[current_assigned]
        ]
        delta_support = (
            edge_support_mask & parent_supported & (current_parent_rows >= 0)
        )
        parent_delta = (
            physical.maplet_centers[truth_parent_rows[delta_support]]
            - physical.maplet_centers[current_parent_rows[delta_support]]
        )
        delta_cluster_rows = (
            np.round(parent_delta / 0.50).astype(np.int64)
            if parent_delta.size else np.zeros((0, 3), dtype=np.int64)
        )
        if delta_cluster_rows.size:
            unique_delta, delta_count = np.unique(
                delta_cluster_rows, axis=0, return_counts=True,
            )
            delta_order = np.argsort(-delta_count, kind="stable")[:8]
            parent_delta_clusters = [
                {
                    "translation_bin_center_m": (
                        0.50 * unique_delta[index]
                    ).tolist(),
                    "support_count": int(delta_count[index]),
                }
                for index in delta_order.tolist()
            ]
        else:
            parent_delta_clusters = []
        oracle_transport = {
            "translation_delta_m": None,
            "assigned_support_count": 0,
            "truth_parent_agreement_fraction": None,
            "pose": None,
        }
        if parent_delta.size:
            transport_delta = np.median(parent_delta, axis=0)
            transport_parent = np.full(current_parent_rows.shape, -1, dtype=np.int64)
            transport_slot = np.full(current_parent_rows.shape, -1, dtype=np.int64)
            parent_row_by_id = {
                int(value): row
                for row, value in enumerate(physical.maplet_ids.tolist())
            }
            for support in range(grouped.xy.shape[0]):
                source_parent = int(current_parent_rows[support])
                if source_parent < 0:
                    continue
                slots = np.flatnonzero(grouped_parent_probability[support] > 0.0)
                if slots.size == 0:
                    continue
                targets = np.asarray([
                    parent_row_by_id.get(
                        int(grouped_parent_ids[support, slot]), -1,
                    )
                    for slot in slots.tolist()
                ], dtype=np.int64)
                valid_target = targets >= 0
                slots, targets = slots[valid_target], targets[valid_target]
                if targets.size == 0:
                    continue
                residual = np.linalg.norm(
                    physical.maplet_centers[targets]
                    - physical.maplet_centers[source_parent][None]
                    - transport_delta[None],
                    axis=1,
                )
                best = int(np.argmin(residual))
                if residual[best] <= 1.0:
                    transport_parent[support] = int(targets[best])
                    transport_slot[support] = int(slots[best])
            transport_children = np.full(current_children.shape, -1, dtype=np.int64)
            for support in np.flatnonzero(transport_slot >= 0).tolist():
                slot = int(transport_slot[support])
                if actual_child.best_child_rows_by_parent is not None:
                    transport_children[support] = int(
                        actual_child.best_child_rows_by_parent[support, slot]
                    )
            transport_mask = edge_support_mask & (transport_children >= 0)
            transport_pose, transport_pose_report = _solve_configuration_pose(
                xy_px,
                transport_children,
                transport_mask,
                physical,
                camera,
                labels.pose_w2c,
            )
            comparable_transport = parent_supported & (transport_parent >= 0)
            oracle_transport = {
                "translation_delta_m": transport_delta.tolist(),
                "assigned_support_count": int(np.sum(transport_mask)),
                "truth_parent_agreement_fraction": (
                    float(np.mean(
                        transport_parent[comparable_transport]
                        == truth_parent_rows[comparable_transport]
                    )) if np.any(comparable_transport) else None
                ),
                "pose": transport_pose_report,
                "pose_available": bool(transport_pose is not None),
            }

        runtime_details = list(
            report_rows["pre_nms"][image_id]["mode_details"][MODE]
        )
        runtime = []
        for source_index, detail in enumerate(runtime_details):
            pose = _pose(detail)
            _, _, children = _score_pose(
                pose, xy_px, extent_px, actual_child, physical, camera,
            )
            common = edge_support_mask & (truth_children >= 0) & (children >= 0)
            truth_common = _configuration_score(
                graph, grouped, truth_children, truth_pose, physical, camera, common,
            )
            wrong_score = _configuration_score(
                graph, grouped, children, pose, physical, camera, common,
            )
            runtime.append({
                "source_index": int(source_index),
                "runtime_rank": int(detail["rank"]),
                "translation_m": float(detail["translation_m"]),
                "rotation_deg": float(detail["rotation_deg"]),
                "supporting_region_count": int(detail["supporting_region_count"]),
                "common_assigned_support_count": int(np.sum(common)),
                "paired_edge_margin": _paired_margin(truth_common, wrong_score),
            })
        runtime.sort(
            key=lambda item: (
                float(item["translation_m"]) / 0.5
                + float(item["rotation_deg"]) / 5.0,
                int(item["runtime_rank"]),
            )
        )
        near_phase = runtime[: int(args.near_phase_candidates)]
        near_phase_margins = [
            item["paired_edge_margin"]["all"]["truth_minus_wrong_margin"]
            for item in near_phase
            if item["paired_edge_margin"]["all"]["truth_minus_wrong_margin"] is not None
        ]
        worst_case_margin = (
            float(min(near_phase_margins)) if near_phase_margins else None
        )
        near_phase_positive = bool(
            worst_case_margin is not None and worst_case_margin > 0.0
        )
        row = {
            "image_id": image_id,
            "support_count": int(grouped.xy.shape[0]),
            "edge_support_count": int(edge_supports.size),
            "edge_counts": {
                "all": int(graph.source.size),
                "local": int(np.sum(graph.kind == 0)),
                "long_range": int(np.sum(graph.kind == 1)),
                "high_displacement": int(np.sum(graph.high_displacement)),
                "distinctive_context": int(np.sum(graph.distinctive_context)),
            },
            "runtime_supported_truth_parent_count": int(np.sum(parent_supported)),
            "truth_configuration_pose": truth_pose_report,
            "naive_same_support_pnp_sanity_control": {
                **naive_truth_pose_report,
                "pose_available": bool(naive_truth_pose is not None),
                "role": (
                    "circularity control only; it is not used for query-edge scoring"
                ),
            },
            "truth_configuration_edge_score": truth_primary,
            "truth_configuration_gt_pose_control": truth_gt_control,
            "current_selected": {
                "runtime_rank": int(current_detail["rank"]),
                "translation_m": float(current_detail["translation_m"]),
                "rotation_deg": float(current_detail["rotation_deg"]),
                "common_assigned_support_count": current_common_count,
                "paired_edge_margin": current_margin,
                "truth_parent_minus_current_parent_translation_clusters": (
                    parent_delta_clusters
                ),
                "oracle_median_translation_transport_sanity_control": (
                    oracle_transport
                ),
            },
            "near_phase_runtime_configurations": near_phase,
            "truth_minus_best_near_phase_wrong_all_edge_margin": worst_case_margin,
            "positive_current_all_edge_margin": bool(
                current_margin["all"]["truth_minus_wrong_margin"] is not None
                and float(current_margin["all"]["truth_minus_wrong_margin"]) > 0.0
            ),
            "positive_near_phase_all_edge_margin": near_phase_positive,
            "clear_near_phase_all_edge_margin": bool(
                worst_case_margin is not None and worst_case_margin > 0.10
            ),
        }
        rows.append(row)
        print(json.dumps({
            "image_id": image_id,
            "truth_pose": truth_pose_report,
            "current_all_edge_margin": current_margin["all"]["truth_minus_wrong_margin"],
            "near_phase_worst_case_margin": worst_case_margin,
        }), flush=True)

    current_positive = int(np.sum([row["positive_current_all_edge_margin"] for row in rows]))
    near_positive = int(np.sum([row["positive_near_phase_all_edge_margin"] for row in rows]))
    clear_near = int(np.sum([row["clear_near_phase_all_edge_margin"] for row in rows]))
    result = {
        "stage": "goal_maplet_query_edge_oracle_factor_audit_g20_4_a",
        "query_count": len(rows),
        "image_ids": list(args.image_ids),
        "physical_map_sha256": physical.content_sha256,
        "canonical_field_sha256": field.content_sha256,
        "surface_mapper_sha256": file_sha256(Path(args.surface_mapper)),
        "physical_instance_readout_sha256": file_sha256(
            Path(args.physical_instance_readout)
        ),
        "feature_pipeline_outer_crossfit": True,
        "phase_used_for_pose_generation": False,
        "ground_truth_usage": (
            "oracle truth-parent diagnostic, pose-error reporting and near-phase "
            "diagnostic identification only; no learned or production ranking"
        ),
        "query_edge_contract": {
            "topology": "candidate_independent_query_local4_long2_v1",
            "coordinate_frame": "normalized_distorted_camera_image_after_pose_projection",
            "primary_factor": "negative_mean_huber_signed_vector_residual_extent_scaled",
            "relative_scale_and_order": "reported_separately_not_weight_fused",
            "clear_margin_threshold_huber_units": 0.10,
        },
        "input_artifacts": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in pool_paths.items()
        },
        "crossfit_evaluation": {
            "path": str(args.crossfit_evaluation),
            "sha256": file_sha256(Path(args.crossfit_evaluation)),
            "heldout_trajectories": crossfit.get("heldout_trajectories", []),
        },
        "summary": {
            "positive_current_all_edge_margin_count": current_positive,
            "positive_near_phase_all_edge_margin_count": near_positive,
            "clear_near_phase_all_edge_margin_count": clear_near,
            "oracle_gate_passed": bool(current_positive >= 2 and near_positive >= 2),
            "gate_definition": (
                "truth beats current selected and each of 8 oracle-identified "
                "near-phase configurations on signed all-edge margin for at least 2/3 frames"
            ),
            "fixed_0p10_margin_sensitivity_only": (
                "reported separately and not used as a model-development gate"
            ),
        },
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"]), flush=True)


if __name__ == "__main__":
    main()
