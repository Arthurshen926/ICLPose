"""Evaluate Top-N region/child coarse pose proposal coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization.alike_detector_only import AlikeDetectorOnly
from feature_extract.vfm.localization.surface_maplet_mapper import load_surface_maplet_mapper
from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalSurfaceField, readout_canonical_field
from feature_extract.vfm.localization_goal_maplet.feature_contract import FieldFeatureContract
from feature_extract.vfm.localization_goal_maplet.child_retrieval import ChildTilePosterior, retrieve_children_given_parents
from feature_extract.vfm.localization_goal_maplet.detector_radio_refiner import refine_pose_with_detector_radio
from feature_extract.vfm.localization_goal_maplet.local_head import load_child_local_head
from feature_extract.vfm.localization_goal_maplet.pfir import (
    ContributorLabels,
    contributor_multiscale_child_distribution,
    contributor_multiscale_maplet_distribution,
)
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.physical_instance_readout import (
    encode_physical_instance_regions,
    load_physical_instance_readout,
    transform_canonical_field_for_role,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.pose_proposal import (
    generate_graph_conditioned_pose_modes,
    generate_parent_then_child_pose_modes,
    generate_region_pose_modes,
    CoarsePoseModes,
)
from feature_extract.vfm.localization_goal_maplet.pose_ranking import (
    merge_cascade_identity_rankings,
    rerank_modes_with_rendered_identity,
)
from feature_extract.vfm.localization_goal_maplet.query_support import (
    aggregate_group_descriptors,
    aggregate_group_posteriors,
    all_token_coordinates,
    group_tokens_after_retrieval,
)
from feature_extract.vfm.localization_goal_maplet.retrieval import ValidityCalibration, retrieve_maplet_posterior
from feature_extract.vfm.localization_goal_maplet.typed_graph import TypedParentGraph
from feature_extract.vfm.query_to_3d_matching import pnp_pose_error
from feature_extract.vfm.surface_maplet_bank import RadioFinalRegionConfig, encode_radio_final_regions


def _camera(path: Path) -> ColmapCamera:
    with np.load(path, allow_pickle=False) as data:
        return ColmapCamera(
            camera_id=0,
            model_id=int(data["camera_model_id"]),
            width=int(data["camera_width"]),
            height=int(data["camera_height"]),
            params=tuple(np.asarray(data["camera_params"], dtype=np.float64).tolist()),
        )


def _pose_report(modes, gt_pose: np.ndarray) -> dict[str, object]:
    errors = [pnp_pose_error(pose, gt_pose) for pose in modes.poses_w2c]
    translation = np.asarray([value.translation_m for value in errors], dtype=np.float64)
    rotation = np.asarray([value.rotation_deg for value in errors], dtype=np.float64)
    report: dict[str, object] = {
        "mode_count": int(len(errors)),
        "current_top1_translation_m": float(translation[0]) if translation.size else None,
        "current_top1_rotation_deg": float(rotation[0]) if rotation.size else None,
    }
    for top_n in (1, 4, 16, 32):
        count = min(int(top_n), translation.size)
        if count == 0:
            report[f"oracle_top{top_n}_translation_m"] = None
            report[f"oracle_top{top_n}_rotation_deg"] = None
            report[f"coverage_top{top_n}_1m_10deg"] = False
            report[f"coverage_top{top_n}_0.5m_5deg"] = False
            continue
        joint = translation[:count] / 0.5 + rotation[:count] / 5.0
        best = int(np.argmin(joint))
        report[f"oracle_top{top_n}_translation_m"] = float(translation[best])
        report[f"oracle_top{top_n}_rotation_deg"] = float(rotation[best])
        report[f"coverage_top{top_n}_1m_10deg"] = bool(np.any((translation[:count] <= 1.0) & (rotation[:count] <= 10.0)))
        report[f"coverage_top{top_n}_0.5m_5deg"] = bool(np.any((translation[:count] <= 0.5) & (rotation[:count] <= 5.0)))
    return report


def _pose_details(modes, gt_pose: np.ndarray) -> list[dict[str, object]]:
    result = []
    for rank, (pose, score, support) in enumerate(
        zip(modes.poses_w2c, modes.scores, modes.supporting_region_count), start=1
    ):
        error = pnp_pose_error(pose, gt_pose)
        center = -pose[:3, :3].T @ pose[:3, 3]
        result.append({
            "rank": int(rank),
            "score": float(score),
            "supporting_region_count": int(support),
            "translation_m": float(error.translation_m),
            "rotation_deg": float(error.rotation_deg),
            "camera_center": center.tolist(),
            "pose_w2c": pose.tolist(),
        })
    return result


def _stable_proposal_seed(image_id: str) -> int:
    """Return a deterministic seed accepted by OpenCV's signed C-int API."""
    digest = hashlib.sha256(str(image_id).encode("utf8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False) & 0x7FFFFFFF


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument("--surface_mapper", required=True)
    parser.add_argument("--field_feature_contract", required=True)
    parser.add_argument("--validity_calibration", required=True)
    parser.add_argument("--typed_graph", default="")
    parser.add_argument("--physical_instance_readout", default="")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--child_local_head", default="")
    parser.add_argument("--parent_mode", choices=("actual", "oracle", "both"), default="both")
    parser.add_argument("--child_mode", choices=("actual", "oracle", "both"), default="both")
    parser.add_argument("--parent_candidates", type=int, default=64)
    parser.add_argument("--child_candidates", type=int, default=64)
    parser.add_argument("--child_temperature", type=float, default=0.07)
    parser.add_argument("--grouping_cosine", type=float, default=0.96)
    parser.add_argument("--maximum_modes", type=int, default=32)
    parser.add_argument("--proposal_trials", type=int, default=2048)
    parser.add_argument("--proposal_method", choices=("random", "graph", "hierarchical"), default="graph")
    parser.add_argument("--graph_seed_parent_pair_count", type=int, default=0)
    parser.add_argument("--local_evidence_weight", type=float, default=0.0)
    parser.add_argument("--render_identity_rerank", action="store_true")
    parser.add_argument("--identity_render_mode", choices=("child_splat", "full_2dgs", "cascade"), default="child_splat")
    parser.add_argument("--cascade_topk", type=int, default=4)
    parser.add_argument("--cascade_disagreement_m", type=float, default=0.75)
    parser.add_argument("--cascade_disagreement_deg", type=float, default=3.0)
    parser.add_argument("--cascade_margin", type=float, default=0.015)
    parser.add_argument("--cascade_always_exact", action="store_true")
    parser.add_argument("--detector_radio_refine_topn", type=int, default=0)
    parser.add_argument("--query_image_root", default="")
    parser.add_argument("--alike_matcha_repo", default="/root/matcha")
    parser.add_argument("--maximum_queries", type=int, default=0)
    parser.add_argument("--image_id", default="")
    parser.add_argument("--include_trajectories", nargs="+", default=None)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite pose-mode report")
    physical = GoalMapletPhysicalMap.load_npz(Path(args.physical_map))
    field = CanonicalSurfaceField.load_npz(Path(args.canonical_field))
    feature_contract = FieldFeatureContract.load_json(Path(args.field_feature_contract))
    if feature_contract.query_readout_type != "surface_maplet_mapper":
        raise ValueError("pose-mode retrieval requires the frozen surface-maplet mapper readout")
    feature_contract.validate(field, query_readout_path=Path(args.surface_mapper))
    readout = readout_canonical_field(field, physical)
    local_field = field
    instance_readout = None
    instance_readout_sha256 = None
    if args.physical_instance_readout:
        instance_readout, instance_metadata = load_physical_instance_readout(
            Path(args.physical_instance_readout), device=str(args.device),
        )
        for key, expected in (
            ("physical_map_sha256", physical.content_sha256),
            ("canonical_field_sha256", field.content_sha256),
        ):
            if instance_metadata.get(key) != expected:
                raise ValueError(f"physical-instance readout lineage differs: {key}")
        readout = type(readout)(
            instance_readout.project_numpy(
                readout.parent_descriptors, role="context", device=str(args.device),
            ),
            readout.parent_coverage,
            instance_readout.project_numpy(
                readout.child_descriptors, role="local", device=str(args.device),
            ),
            readout.child_coverage,
        )
        local_field = transform_canonical_field_for_role(
            instance_readout, field, role="local", device=str(args.device),
        )
        instance_readout_sha256 = file_sha256(Path(args.physical_instance_readout))
    calibration = ValidityCalibration.load_json(Path(args.validity_calibration))
    mapper, _ = load_surface_maplet_mapper(Path(args.surface_mapper), device=str(args.device))
    graph = None
    if args.proposal_method in ("graph", "hierarchical"):
        if not args.typed_graph:
            raise ValueError("graph proposal requires --typed_graph")
        graph = TypedParentGraph.load_npz(Path(args.typed_graph))
        if graph.physical_map_sha256 != physical.content_sha256 or graph.canonical_field_sha256 != field.content_sha256:
            raise ValueError("typed graph lineage differs")
        graph_readout = graph.metadata.get("physical_instance_readout_sha256")
        if graph_readout != instance_readout_sha256:
            raise ValueError("typed graph physical-instance readout lineage differs")
    detector = None
    if int(args.detector_radio_refine_topn) > 0:
        if not args.query_image_root:
            raise ValueError("detector-radio refinement requires --query_image_root")
        detector = AlikeDetectorOnly(
            device=str(args.device), matcha_repo=Path(args.alike_matcha_repo)
        )
    child_map_descriptor = readout.child_descriptors
    local_head = None
    if args.child_local_head:
        artifact = load_child_local_head(Path(args.child_local_head), device=str(args.device))
        if artifact.metadata.get("physical_map_sha256") != physical.content_sha256 or artifact.metadata.get("canonical_field_sha256") != field.content_sha256:
            raise ValueError("child-local head lineage differs")
        local_head = artifact.model
        with torch.no_grad():
            child_map_descriptor = local_head.encode_map(
                torch.as_tensor(child_map_descriptor, dtype=torch.float32, device=str(args.device))
            ).cpu().numpy()
    paths = sorted(Path(args.contributors).glob("*.npz"))[int(args.shard_index) :: int(args.shard_count)]
    if args.image_id:
        selected_paths = []
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                item = json.loads(str(np.asarray(data["metadata_json"]).item()))
            if str(item.get("image_id", "")) == str(args.image_id):
                selected_paths.append(path)
        paths = selected_paths
    if args.include_trajectories:
        requested = set(str(value) for value in args.include_trajectories)
        selected_paths = []
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                item = json.loads(str(np.asarray(data["metadata_json"]).item()))
            if str(item.get("trajectory_id", str(item.get("image_id", "")).split("/", 1)[0])) in requested:
                selected_paths.append(path)
        paths = selected_paths
    if int(args.maximum_queries) > 0:
        paths = paths[: int(args.maximum_queries)]
    reports = []
    context_config = RadioFinalRegionConfig()
    local_config = RadioFinalRegionConfig(pool_sizes=(1,), pool_weights=(1.0,))
    for path in paths:
        labels = ContributorLabels.load_npz(path)
        camera = _camera(path)
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        with np.load(Path(str(metadata["token_path"])), allow_pickle=False) as data:
            raw = np.asarray(data["radio_final"], dtype=np.float32)
        mapped = mapper.project(raw).measurement_context
        _, token_xy = all_token_coordinates(int(raw.shape[1]), int(raw.shape[2]))
        if instance_readout is None:
            context_descriptor = encode_radio_final_regions(mapped, token_xy, context_config)
            local_descriptor = encode_radio_final_regions(mapped, token_xy, local_config)
        else:
            context_descriptor = encode_physical_instance_regions(
                instance_readout, mapped, token_xy, role="context", device=str(args.device),
            )
            local_descriptor = encode_physical_instance_regions(
                instance_readout, mapped, token_xy, role="local", device=str(args.device),
            )
        if local_head is not None:
            with torch.no_grad():
                local_descriptor = local_head.encode_query(
                    torch.as_tensor(local_descriptor, dtype=torch.float32, device=str(args.device))
                ).cpu().numpy()
        parent_ids, parent_probability, parent_null, _ = retrieve_maplet_posterior(
            context_descriptor, readout.parent_descriptors, physical.maplet_ids,
            readout.parent_coverage > 0.0,
            maximum_candidates=int(args.parent_candidates), temperature=0.07,
            null_similarity_center=float(calibration.center), null_similarity_scale=float(calibration.scale),
        )
        grouped = group_tokens_after_retrieval(
            token_xy, context_descriptor, parent_ids[:, 0],
            token_height=int(raw.shape[1]), token_width=int(raw.shape[2]),
            image_width=int(camera.width), image_height=int(camera.height),
            descriptor_half_size_tokens=2.0,
            minimum_descriptor_cosine=float(args.grouping_cosine),
        )
        grouped_parent_ids, grouped_parent_probability, grouped_parent_null = aggregate_group_posteriors(
            parent_ids, parent_probability, parent_null,
            grouped.member_offsets, grouped.member_token_indices,
            maximum_candidates=int(args.parent_candidates),
        )
        grouped_local = aggregate_group_descriptors(local_descriptor, grouped.member_offsets, grouped.member_token_indices)
        parent_inputs = {}
        truth_child = truth_child_null = None
        if args.parent_mode in ("actual", "both"):
            parent_inputs["actual_parent"] = (grouped_parent_ids, grouped_parent_probability, grouped_parent_null)
        if args.parent_mode in ("oracle", "both"):
            truth, truth_null = contributor_multiscale_maplet_distribution(
                labels, physical, token_xy,
                token_height=int(raw.shape[1]), token_width=int(raw.shape[2]),
                pool_sizes=(1,), pool_weights=(1.0,),
                group_member_offsets=grouped.member_offsets,
                group_member_token_indices=grouped.member_token_indices,
            )
            parent_row = np.argmax(truth, axis=1)
            parent_inputs["oracle_parent"] = (
                physical.maplet_ids[parent_row, None], (1.0 - truth_null)[:, None], truth_null,
            )
        if args.child_mode in ("oracle", "both"):
            truth_child, truth_child_null = contributor_multiscale_child_distribution(
                labels, physical, token_xy,
                token_height=int(raw.shape[1]), token_width=int(raw.shape[2]),
                group_member_offsets=grouped.member_offsets,
                group_member_token_indices=grouped.member_token_indices,
            )
        xy_px = grouped.xy * np.asarray([camera.width, camera.height], dtype=np.float32)
        extent_px = grouped.extent * np.asarray([camera.width, camera.height], dtype=np.float32)
        modes_report = {}
        mode_details = {}
        ranking_diagnostics = {}
        detected = None
        if detector is not None:
            detected = detector.detect(
                Path(args.query_image_root) / str(metadata["image_id"]),
                image_width=int(camera.width), image_height=int(camera.height),
                top_k=512, candidate_top_k=4096,
            )
        child_inputs = {}
        if args.child_mode in ("actual", "both"):
            for name, values in parent_inputs.items():
                child_inputs[f"{name}_actual_child"] = (
                    retrieve_children_given_parents(
                        grouped_local, *values, child_map_descriptor, readout.child_coverage, physical,
                        maximum_child_candidates=int(args.child_candidates),
                        temperature=float(args.child_temperature),
                    ),
                    values,
                )
        if args.child_mode in ("oracle", "both"):
            keep = min(int(args.child_candidates), truth_child.shape[1])
            columns = np.argpartition(-truth_child, kth=keep - 1, axis=1)[:, :keep]
            order = np.argsort(-np.take_along_axis(truth_child, columns, axis=1), axis=1, kind="stable")
            columns = np.take_along_axis(columns, order, axis=1)
            probability = np.take_along_axis(truth_child, columns, axis=1)
            oracle_parent_values = parent_inputs.get("oracle_parent", next(iter(parent_inputs.values())))
            child_inputs["oracle_child"] = (ChildTilePosterior(columns, probability, truth_child_null), oracle_parent_values)
        for name, (child, parent_values) in child_inputs.items():
            stable_seed = _stable_proposal_seed(str(metadata["image_id"]))
            if args.proposal_method in ("graph", "hierarchical"):
                proposal = (
                    generate_parent_then_child_pose_modes
                    if args.proposal_method == "hierarchical"
                    else generate_graph_conditioned_pose_modes
                )
                modes = proposal(
                    xy_px, extent_px, *parent_values, child, physical, graph, camera,
                    maximum_modes=int(args.maximum_modes),
                    random_seed=stable_seed,
                    **(
                        {
                            "local_evidence_weight": float(args.local_evidence_weight),
                            "seed_parent_pair_count": int(args.graph_seed_parent_pair_count),
                        }
                        if args.proposal_method == "graph" else {}
                    ),
                )
            else:
                modes = generate_region_pose_modes(
                    xy_px, extent_px, child, physical, camera,
                    maximum_modes=int(args.maximum_modes), proposal_trials=int(args.proposal_trials),
                    random_seed=stable_seed,
                )
            if args.render_identity_rerank and modes.poses_w2c.shape[0]:
                base_modes = modes
                ranking = rerank_modes_with_rendered_identity(
                    modes,
                    grouped.member_offsets,
                    grouped.member_token_indices,
                    *parent_values,
                    child,
                    physical,
                    local_field,
                    camera,
                    token_height=int(raw.shape[1]),
                    token_width=int(raw.shape[2]),
                    render_mode=("child_splat" if args.identity_render_mode == "cascade" else str(args.identity_render_mode)),
                    device=str(args.device),
                )
                modes = ranking.modes
                ranking_diagnostics[name] = {
                    "identity_scores": ranking.identity_scores.tolist(),
                    "parent_log_likelihood": ranking.parent_log_likelihood.tolist(),
                    "child_log_likelihood": ranking.child_log_likelihood.tolist(),
                    "rendered_coverage": ranking.rendered_coverage.tolist(),
                    "proposal_scores": ranking.proposal_scores.tolist(),
                    "original_indices": ranking.original_indices.tolist(),
                }
                if args.identity_render_mode == "cascade" and modes.poses_w2c.shape[0]:
                    base_center = -base_modes.poses_w2c[0, :3, :3].T @ base_modes.poses_w2c[0, :3, 3]
                    splat_center = -modes.poses_w2c[0, :3, :3].T @ modes.poses_w2c[0, :3, 3]
                    translation_disagreement = float(np.linalg.norm(base_center - splat_center))
                    relative = base_modes.poses_w2c[0, :3, :3] @ modes.poses_w2c[0, :3, :3].T
                    rotation_disagreement = float(np.degrees(np.arccos(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))))
                    splat_margin = (
                        float(ranking.identity_scores[0] - ranking.identity_scores[1])
                        if ranking.identity_scores.size > 1 else float("inf")
                    )
                    gate = bool(
                        bool(args.cascade_always_exact)
                        or
                        translation_disagreement >= float(args.cascade_disagreement_m)
                        or rotation_disagreement >= float(args.cascade_disagreement_deg)
                        or splat_margin <= float(args.cascade_margin)
                    )
                    ranking_diagnostics[name]["cascade_gate"] = gate
                    ranking_diagnostics[name]["base_splat_translation_disagreement_m"] = translation_disagreement
                    ranking_diagnostics[name]["base_splat_rotation_disagreement_deg"] = rotation_disagreement
                    ranking_diagnostics[name]["splat_margin"] = splat_margin
                    if gate:
                        take = min(int(args.cascade_topk), modes.poses_w2c.shape[0])
                        subset = CoarsePoseModes(
                            modes.poses_w2c[:take], modes.scores[:take], modes.supporting_region_count[:take]
                        )
                        exact = rerank_modes_with_rendered_identity(
                            subset,
                            grouped.member_offsets,
                            grouped.member_token_indices,
                            *parent_values,
                            child,
                            physical,
                            local_field,
                            camera,
                            token_height=int(raw.shape[1]),
                            token_width=int(raw.shape[2]),
                            render_mode="full_2dgs",
                            device=str(args.device),
                        )
                        cascade = merge_cascade_identity_rankings(
                            ranking, exact, exact_candidate_count=take
                        )
                        modes = cascade.modes
                        ranking_diagnostics[name].update({
                            "identity_scores": cascade.cheap_identity_scores.tolist(),
                            "parent_log_likelihood": cascade.cheap_parent_log_likelihood.tolist(),
                            "child_log_likelihood": cascade.cheap_child_log_likelihood.tolist(),
                            "rendered_coverage": cascade.cheap_rendered_coverage.tolist(),
                            "proposal_scores": cascade.proposal_scores.tolist(),
                            "original_indices": cascade.original_indices.tolist(),
                            "cascade_exact_evaluated": cascade.exact_evaluated.tolist(),
                            "cascade_exact_scores": cascade.exact_identity_scores.tolist(),
                            "cascade_exact_parent_log_likelihood": cascade.exact_parent_log_likelihood.tolist(),
                            "cascade_exact_child_log_likelihood": cascade.exact_child_log_likelihood.tolist(),
                            "cascade_exact_rendered_coverage": cascade.exact_rendered_coverage.tolist(),
                        })
            modes_report[name] = _pose_report(modes, labels.pose_w2c)
            mode_details[name] = _pose_details(modes, labels.pose_w2c)
            if detected is not None and modes.poses_w2c.shape[0]:
                refined = []
                for pose in modes.poses_w2c[: int(args.detector_radio_refine_topn)]:
                    value = refine_pose_with_detector_radio(
                        mapped,
                        detected.xy,
                        detected.scores,
                        xy_px,
                        extent_px,
                        child,
                        pose,
                        physical,
                        local_field,
                        camera,
                    )
                    if value.success:
                        quality = (
                            float(value.inlier_count / max(value.match_count, 1))
                            + 0.002 * float(value.inlier_count)
                            - 0.01 * float(value.median_reprojection_px)
                            + 0.10 * float(value.mean_inlier_similarity)
                        )
                        refined.append((quality, value))
                refined.sort(key=lambda item: (-item[0], -item[1].inlier_count))
                if refined:
                    refined_modes = CoarsePoseModes(
                        np.asarray([item[1].pose_w2c for item in refined]),
                        np.asarray([item[0] for item in refined]),
                        np.asarray([item[1].inlier_count for item in refined]),
                    )
                    refined_name = f"{name}_detector_radio"
                    modes_report[refined_name] = _pose_report(refined_modes, labels.pose_w2c)
                    mode_details[refined_name] = [
                        {
                            **detail,
                            "selected_child_count": int(item[1].selected_child_count),
                            "candidate_primitive_count": int(item[1].candidate_primitive_count),
                            "match_count": int(item[1].match_count),
                            "inlier_count": int(item[1].inlier_count),
                            "median_reprojection_px": float(item[1].median_reprojection_px),
                            "mean_inlier_similarity": float(item[1].mean_inlier_similarity),
                        }
                        for detail, item in zip(_pose_details(refined_modes, labels.pose_w2c), refined)
                    ]
        report = {
            "image_id": str(metadata["image_id"]),
            "support_count": int(grouped.xy.shape[0]),
            "modes": modes_report,
            "mode_details": mode_details,
            "ranking_diagnostics": ranking_diagnostics,
        }
        reports.append(report)
        print(json.dumps(report), flush=True)
    mode_names = sorted({name for row in reports for name in row["modes"]})
    metric_names = sorted({name for row in reports for mode in row["modes"].values() for name in mode})
    summary = {}
    for mode in mode_names:
        summary[mode] = {}
        for metric in metric_names:
            values = [row["modes"][mode].get(metric) for row in reports if mode in row["modes"]]
            values = [value for value in values if value is not None]
            if not values:
                summary[mode][metric] = None
            elif isinstance(values[0], bool):
                summary[mode][metric] = float(np.mean(values))
            else:
                summary[mode][metric] = float(np.median(values))
    result = {
        "stage": "goal_maplet_region_child_topn_pose_proposal",
        "query_count": len(reports),
        "physical_map_sha256": physical.content_sha256,
        "canonical_field_sha256": field.content_sha256,
        "field_feature_contract_sha256": feature_contract.content_sha256,
        "validity_calibration_sha256": calibration.content_sha256,
        "parent_mode": str(args.parent_mode),
        "child_mode": str(args.child_mode),
        "proposal_trials": int(args.proposal_trials),
        "proposal_method": str(args.proposal_method),
        "graph_seed_parent_pair_count": (
            int(args.graph_seed_parent_pair_count) if args.proposal_method == "graph" else None
        ),
        "proposal_seed_policy": "sha256_image_id_uint31_little_endian_v1",
        "local_evidence_weight": float(args.local_evidence_weight),
        "render_identity_rerank": bool(args.render_identity_rerank),
        "identity_render_mode": str(args.identity_render_mode),
        "cascade_contract": {
            "topk": int(args.cascade_topk),
            "disagreement_m": float(args.cascade_disagreement_m),
            "disagreement_deg": float(args.cascade_disagreement_deg),
            "splat_margin": float(args.cascade_margin),
            "always_exact": bool(args.cascade_always_exact),
        },
        "detector_radio_refine_topn": int(args.detector_radio_refine_topn),
        "alike_detector_only": bool(detector is not None),
        "typed_graph_sha256": graph.content_sha256 if graph is not None else None,
        "physical_instance_readout_sha256": instance_readout_sha256,
        "maximum_modes": int(args.maximum_modes),
        "summary": summary,
        "rows": reports,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
