"""Verify frozen Goal-Maplet pose modes by exact canonical-field rendering.

This is a pose-level VFM likelihood, not a point matcher.  Each frozen pose
renders the complete clean 2DGS scene from the single stored canonical RADIO
field.  The rendered field is compared with the runtime query on a fixed image
grid, so a pose cannot improve its score by hiding difficult regions.  Offline
teacher features are represented only through the regenerable G17 role head;
no teacher embedding or mapping image is loaded at deployment.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.localization.surface_maplet_mapper import load_surface_maplet_mapper
from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalSurfaceField
from feature_extract.vfm.localization_goal_maplet.feature_contract import FieldFeatureContract
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.physical_instance_readout import (
    encode_physical_instance_regions,
    load_physical_instance_readout,
    transform_canonical_field_for_role,
)
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.pfir import ContributorLabels
from feature_extract.vfm.localization_goal_maplet.surface_refiner import canonical_alignment_score
from feature_extract.vfm.localization_goal_maplet.surface_renderer import render_canonical_surface_field
from feature_extract.vfm.colmap_tracks import ColmapCamera


def _camera(path: Path) -> ColmapCamera:
    with np.load(path, allow_pickle=False) as data:
        return ColmapCamera(
            0,
            int(data["camera_model_id"]),
            int(data["camera_width"]),
            int(data["camera_height"]),
            tuple(np.asarray(data["camera_params"], dtype=np.float64).tolist()),
        )


def _pose_key(value: object) -> tuple[float, ...]:
    return tuple(np.asarray(value, dtype=np.float64).round(9).reshape(-1).tolist())


def _summary(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    names = sorted({name for row in rows for name in row.get("modes", {})})
    result: dict[str, dict[str, object]] = {}
    for name in names:
        selected = [row["modes"][name] for row in rows if name in row.get("modes", {})]
        metrics = sorted({key for item in selected for key in item})
        result[name] = {}
        for metric in metrics:
            values = [item.get(metric) for item in selected if item.get(metric) is not None]
            if not values:
                result[name][metric] = None
            elif isinstance(values[0], bool):
                result[name][metric] = float(np.mean(values))
            elif metric == "mode_count":
                result[name][metric] = float(np.mean(values))
            else:
                value = np.asarray(values, dtype=np.float64)
                result[name][metric] = {
                    "median": float(np.median(value)),
                    "p90": float(np.percentile(value, 90.0)),
                    "p95": float(np.percentile(value, 95.0)),
                }
    return result


def _pose_report(details: list[dict[str, object]]) -> dict[str, object]:
    if not details:
        return {
            "mode_count": 0,
            "current_top1_translation_m": None,
            "current_top1_rotation_deg": None,
            "oracle_top1_translation_m": None,
            "oracle_top1_rotation_deg": None,
            "oracle_top4_translation_m": None,
            "oracle_top4_rotation_deg": None,
            "oracle_top16_translation_m": None,
            "oracle_top16_rotation_deg": None,
            "oracle_top32_translation_m": None,
            "oracle_top32_rotation_deg": None,
            "coverage_top1_1m_10deg": False,
            "coverage_top1_0.5m_5deg": False,
            "coverage_top4_1m_10deg": False,
            "coverage_top4_0.5m_5deg": False,
            "coverage_top16_1m_10deg": False,
            "coverage_top16_0.5m_5deg": False,
            "coverage_top32_1m_10deg": False,
            "coverage_top32_0.5m_5deg": False,
        }
    translation = np.asarray([item["translation_m"] for item in details], dtype=np.float64)
    rotation = np.asarray([item["rotation_deg"] for item in details], dtype=np.float64)
    report: dict[str, object] = {
        "mode_count": int(len(details)),
        "current_top1_translation_m": float(translation[0]),
        "current_top1_rotation_deg": float(rotation[0]),
    }
    for take in (1, 4, 16, 32):
        end = min(take, len(details))
        quality = translation[:end] + 0.02 * rotation[:end]
        best = int(np.argmin(quality))
        report[f"oracle_top{take}_translation_m"] = float(translation[best])
        report[f"oracle_top{take}_rotation_deg"] = float(rotation[best])
        report[f"coverage_top{take}_1m_10deg"] = bool(
            np.any((translation[:end] <= 1.0) & (rotation[:end] <= 10.0))
        )
        report[f"coverage_top{take}_0.5m_5deg"] = bool(
            np.any((translation[:end] <= 0.5) & (rotation[:end] <= 5.0))
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--candidate_pool", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument("--surface_mapper", required=True)
    parser.add_argument("--field_feature_contract", required=True)
    parser.add_argument("--physical_instance_readout", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--maximum_modes", type=int, default=16)
    parser.add_argument("--role", choices=("local", "context"), default="local")
    parser.add_argument("--spatial_role_readout", action="store_true")
    parser.add_argument("--resume_scores", default="")
    parser.add_argument("--base_maximum_modes", type=int, default=0)
    parser.add_argument("--additional_candidate_pool", default="")
    parser.add_argument("--additional_selection_report", default="")
    parser.add_argument("--additional_selection_policy", default="relation_v2_node_fit")
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite surface verification")
    if not 0 <= int(args.shard_index) < int(args.shard_count):
        raise ValueError("invalid surface-verification shard")

    physical = GoalMapletPhysicalMap.load_npz(Path(args.physical_map))
    field = CanonicalSurfaceField.load_npz(Path(args.canonical_field))
    contract = FieldFeatureContract.load_json(Path(args.field_feature_contract))
    contract.validate(field, query_readout_path=Path(args.surface_mapper))
    mapper, _ = load_surface_maplet_mapper(Path(args.surface_mapper), device=str(args.device))
    readout, readout_metadata = load_physical_instance_readout(
        Path(args.physical_instance_readout), device=str(args.device),
    )
    if str(readout_metadata.get("physical_map_sha256", "")) != physical.content_sha256:
        raise ValueError("physical-instance readout and physical map differ")
    if str(readout_metadata.get("canonical_field_sha256", "")) != field.content_sha256:
        raise ValueError("physical-instance readout and canonical field differ")
    role_field = (
        field
        if bool(args.spatial_role_readout)
        else transform_canonical_field_for_role(
            readout, field, role=str(args.role), device=str(args.device),
        )
    )
    pool_path = Path(args.candidate_pool)
    pool = json.loads(pool_path.read_text())
    expected_readout = file_sha256(Path(args.physical_instance_readout))
    if str(pool.get("physical_map_sha256", "")) != physical.content_sha256:
        raise ValueError("candidate pool and physical map differ")
    if str(pool.get("canonical_field_sha256", "")) != field.content_sha256:
        raise ValueError("candidate pool and canonical field differ")
    if str(pool.get("physical_instance_readout_sha256", "")) != expected_readout:
        raise ValueError("candidate pool and physical-instance readout differ")
    additional_by_image: dict[str, dict[str, dict[str, object]]] = {}
    additional_contract = None
    if bool(args.additional_candidate_pool) != bool(args.additional_selection_report):
        raise ValueError("additional candidate pool and selection report must be provided together")
    if args.additional_candidate_pool:
        additional_pool_path = Path(args.additional_candidate_pool)
        selection_path = Path(args.additional_selection_report)
        additional_pool = json.loads(additional_pool_path.read_text())
        selection = json.loads(selection_path.read_text())
        if str(additional_pool.get("physical_map_sha256", "")) != physical.content_sha256:
            raise ValueError("additional candidate pool and physical map differ")
        if str(additional_pool.get("canonical_field_sha256", "")) != field.content_sha256:
            raise ValueError("additional candidate pool and canonical field differ")
        pool_rows = {str(row["image_id"]): row for row in additional_pool.get("rows", [])}
        for selected in selection.get("rows", []):
            image_id = str(selected["image_id"])
            source = pool_rows.get(image_id)
            policy = selected.get(str(args.additional_selection_policy))
            if source is None or not isinstance(policy, dict) or int(policy.get("rank", 0)) <= 0:
                raise ValueError(f"missing additional selected pose: {image_id}")
            rank = int(policy["rank"]) - 1
            additional_by_image[image_id] = {}
            for name, details in source.get("mode_details", {}).items():
                if rank >= len(details):
                    raise ValueError(f"additional selected pose rank is outside pool: {image_id}")
                value = json.loads(json.dumps(details[rank]))
                value["proposal_source"] = "additional_frozen_geometric_policy"
                additional_by_image[image_id][name] = value
        additional_contract = {
            "candidate_pool_sha256": file_sha256(additional_pool_path),
            "selection_report_sha256": file_sha256(selection_path),
            "selection_policy": str(args.additional_selection_policy),
            "selection_uses_gt_at_runtime": False,
        }
    resumed_by_image: dict[str, dict[tuple[float, ...], tuple[float, float]]] = {}
    resume_sha256 = None
    if args.resume_scores:
        resume_path = Path(args.resume_scores)
        resume = json.loads(resume_path.read_text())
        resume_contract = dict(resume.get("surface_verification_contract") or {})
        expected_contract = {
            "candidate_pool_sha256": file_sha256(pool_path),
            "physical_instance_readout_sha256": expected_readout,
            "role": str(args.role),
            "spatial_role_readout": bool(args.spatial_role_readout),
        }
        for key, value in expected_contract.items():
            if resume_contract.get(key) != value:
                raise ValueError(f"resumed surface scores differ: {key}")
        for row in resume.get("rows", []):
            values: dict[tuple[float, ...], tuple[float, float]] = {}
            for name, details in row.get("mode_details", {}).items():
                diagnostics = row.get("ranking_diagnostics", {}).get(name, {})
                coverage_preorder = diagnostics.get("surface_alignment_coverage_preorder", [])
                original = diagnostics.get("surface_alignment_original_indices", [])
                for rank, detail in enumerate(details):
                    score = detail.get("surface_alignment_score")
                    if score is None:
                        continue
                    source_index = int(original[rank]) if rank < len(original) else -1
                    # Coverage is pose-aligned below using the diagnostic
                    # pre-order when possible.  A resumed score without an
                    # aligned coverage fails closed instead of silently using
                    # a different fixed denominator.
                    if source_index < 0 or source_index >= len(coverage_preorder):
                        continue
                    values[_pose_key(detail["pose_w2c"])] = (
                        float(score), float(coverage_preorder[source_index]),
                    )
            resumed_by_image[str(row["image_id"])] = values
        resume_sha256 = file_sha256(resume_path)

    contributor_by_image: dict[str, Path] = {}
    for path in sorted(Path(args.contributors).glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        image_id = str(metadata["image_id"])
        if image_id in contributor_by_image:
            raise ValueError(f"duplicate contributor image: {image_id}")
        contributor_by_image[image_id] = path

    rows: list[dict[str, object]] = []
    source_rows = sorted(pool.get("rows", []), key=lambda item: str(item["image_id"]))
    source_rows = source_rows[int(args.shard_index) :: int(args.shard_count)]
    for source in source_rows:
        image_id = str(source["image_id"])
        contributor = contributor_by_image.get(image_id)
        if contributor is None:
            raise ValueError(f"missing contributor for candidate query: {image_id}")
        _ = ContributorLabels.load_npz(contributor)  # fail closed on evaluation lineage
        camera = _camera(contributor)
        with np.load(contributor, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        with np.load(Path(str(metadata["token_path"])), allow_pickle=False) as data:
            raw = np.asarray(data["radio_final"], dtype=np.float32)
        mapped = mapper.project(raw).measurement_context
        grid_y, grid_x = np.mgrid[: mapped.shape[1], : mapped.shape[2]]
        token_xy = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1)
        if bool(args.spatial_role_readout):
            query_flat = encode_physical_instance_regions(
                readout,
                mapped,
                token_xy,
                role=str(args.role),
                device=str(args.device),
            )
        else:
            flat = mapped.transpose(1, 2, 0).reshape(-1, mapped.shape[0])
            query_flat = readout.project_numpy(
                flat, role=str(args.role), device=str(args.device),
            )
        query = query_flat.reshape(
            mapped.shape[1], mapped.shape[2], mapped.shape[0],
        ).transpose(2, 0, 1)

        row = json.loads(json.dumps(source))
        for name, details in row.get("mode_details", {}).items():
            if int(args.base_maximum_modes) > 0:
                details = details[: int(args.base_maximum_modes)]
            additional = additional_by_image.get(image_id, {}).get(name)
            if additional is not None and _pose_key(additional["pose_w2c"]) not in {
                _pose_key(detail["pose_w2c"]) for detail in details
            }:
                details.append(additional)
            take = min(max(int(args.maximum_modes), 0), len(details))
            scores: list[float] = []
            coverages: list[float] = []
            for detail in details[:take]:
                resumed = resumed_by_image.get(image_id, {}).get(_pose_key(detail["pose_w2c"]))
                if resumed is not None:
                    scores.append(float(resumed[0]))
                    coverages.append(float(resumed[1]))
                    continue
                rendered = render_canonical_surface_field(
                    physical,
                    role_field,
                    np.asarray(detail["pose_w2c"], dtype=np.float64),
                    camera,
                    width=int(query.shape[2]),
                    height=int(query.shape[1]),
                    device=str(args.device),
                )
                if bool(args.spatial_role_readout):
                    rendered_flat = encode_physical_instance_regions(
                        readout,
                        np.asarray(rendered.feature, dtype=np.float32),
                        token_xy,
                        role=str(args.role),
                        device=str(args.device),
                        spatial_valid_mask=np.asarray(rendered.mask, dtype=bool),
                    )
                    rendered_feature = rendered_flat.reshape(
                        mapped.shape[1], mapped.shape[2], mapped.shape[0],
                    ).transpose(2, 0, 1)
                    rendered = replace(rendered, feature=rendered_feature)
                scores.append(float(canonical_alignment_score(rendered, query)))
                coverages.append(float(np.mean(np.asarray(rendered.mask, dtype=bool))))
            order = sorted(range(take), key=lambda index: (-scores[index], index))
            order.extend(range(take, len(details)))
            reordered = [details[index] for index in order]
            for rank, detail in enumerate(reordered, start=1):
                detail["rank"] = int(rank)
                detail["pre_surface_score"] = float(detail["score"])
                source_index = int(order[rank - 1])
                detail["surface_alignment_score"] = (
                    float(scores[source_index]) if source_index < take else None
                )
                if source_index < take:
                    detail["score"] = float(scores[source_index])
            row["mode_details"][name] = reordered
            row["modes"][name] = _pose_report(reordered)
            diagnostics = row.setdefault("ranking_diagnostics", {}).setdefault(name, {})
            diagnostics["surface_alignment_role"] = str(args.role)
            diagnostics["surface_alignment_evaluated_count"] = int(take)
            diagnostics["surface_alignment_scores_preorder"] = scores
            diagnostics["surface_alignment_coverage_preorder"] = coverages
            diagnostics["surface_alignment_original_indices"] = order
        rows.append(row)
        print(json.dumps({"image_id": image_id, "surface_verified_modes": {
            name: min(int(args.maximum_modes), len(value))
            for name, value in row.get("mode_details", {}).items()
        }}), flush=True)

    verification_contract = {
        "candidate_pool_sha256": file_sha256(pool_path),
        "physical_instance_readout_sha256": expected_readout,
        "role": str(args.role),
        "spatial_role_readout": bool(args.spatial_role_readout),
        "maximum_modes": int(args.maximum_modes),
        "render": "complete_clean_2dgs_single_canonical_field",
        "score": "fixed_full_query_grid_mean_cosine",
        "candidate_dependent_surface_subset": False,
        "stored_map_feature_type_count": 1,
        "stored_downstream_embedding_count": 0,
        "stores_mapping_rgb": False,
        "uses_point_correspondences": False,
        "uses_pnp": False,
        "resumed_score_artifact_sha256": resume_sha256,
        "base_maximum_modes": int(args.base_maximum_modes),
        "additional_candidate_contract": additional_contract,
    }
    result = {
        **{key: value for key, value in pool.items() if key not in ("rows", "summary", "query_count", "shard_count", "source_shards")},
        "stage": "goal_maplet_exact_surface_verified_pose_modes",
        "query_count": len(rows),
        "shard_count": int(args.shard_count),
        "surface_verification_contract": verification_contract,
        "summary": _summary(rows),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
