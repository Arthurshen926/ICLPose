"""Evaluate candidate poses with the direct full-token canonical field renderer.

This path preserves full-scene 2DGS occlusion but bypasses child/parent Top-L
construction.  It renders one canonical RADIO feature and one payload mass per
token, then applies an explicitly selected phase score.  It uses no keypoints,
point correspondences, PnP, or absolute pose regression.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from feature_extract.tools.vfm.build_goal_maplet_sparse_pose_transport_dataset import (
    _load_contributors,
)
from feature_extract.tools.vfm.train_evaluate_goal_maplet_sparse_pose_transport import _metrics
from feature_extract.tools.vfm.verify_goal_maplet_pose_modes_with_surface_field import _camera
from feature_extract.vfm.localization.surface_maplet_mapper import load_surface_maplet_mapper
from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalSurfaceField
from feature_extract.vfm.localization_goal_maplet.fulltoken_surface_pose_energy import (
    CONDITIONAL_SHIFT_PHASE_CONTROL_SEMANTICS,
    SHIFT_TOLERANT_MAXMIN_PHASE_SEMANTICS,
    SHIFT_TOLERANT_PHASE_SEMANTICS,
    conditional_fulltoken_phase_pose_energy_control,
    conservative_fulltoken_phase_pose_energy,
)
from feature_extract.vfm.localization_goal_maplet.fulltoken_pose_ranking import (
    FULLTOKEN_POSE_RANKING_CHANNELS,
    FULLTOKEN_POSE_RANKING_FEATURE_SEMANTICS,
    compact_fulltoken_pose_ranking_features,
    continuous_seed_domain_recall_metrics,
    distinct_pose_basin_recall_metrics,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.pose_candidate_dataset import (
    load_pose_candidate_dataset,
)
from feature_extract.vfm.localization_goal_maplet.resident_surface_renderer import (
    FrozenSoftSurfaceSceneGPU,
)


SCHEMA = "goal_maplet_direct_canonical_fulltoken_pose_energy_evaluation_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument("--surface_mapper", required=True)
    parser.add_argument("--field_feature_contract", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--score_semantics",
        choices=(
            "conditional_phase_shift_control",
            "conservative_phase_shift",
            "conservative_phase_shift_maxmin",
        ),
        default="conditional_phase_shift_control",
    )
    parser.add_argument("--phase_shift_radius_tokens", type=int, default=1)
    parser.add_argument("--render_batch_size", type=int, default=16)
    parser.add_argument("--query_start", type=int, default=0)
    parser.add_argument("--maximum_queries", type=int)
    parser.add_argument("--train_queries", type=int, default=0)
    parser.add_argument(
        "--compact_feature_output",
        help="optional float16 .npy full-layout feature artifact",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not 0 <= int(args.phase_shift_radius_tokens) <= 4:
        raise ValueError("phase shift radius must lie in [0,4]")
    if int(args.render_batch_size) <= 0:
        raise ValueError("render batch size must be positive")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("refusing to overwrite direct canonical evaluation")

    dataset_path = Path(args.dataset)
    arrays, metadata = load_pose_candidate_dataset(
        dataset_path, require_rendered_targets=False,
    )
    physical = GoalMapletPhysicalMap.load_npz(Path(args.physical_map))
    field = CanonicalSurfaceField.load_npz(Path(args.canonical_field))
    contract = json.loads(Path(args.field_feature_contract).read_text())
    mapper_sha = file_sha256(Path(args.surface_mapper))
    if (
        physical.content_sha256 != metadata.get("physical_map_sha256")
        or field.content_sha256 != metadata.get("canonical_field_sha256")
        or field.physical_map_sha256 != physical.content_sha256
        or contract.get("canonical_field_sha256") != field.content_sha256
        or contract.get("query_readout_sha256") != mapper_sha
    ):
        raise ValueError("direct canonical dataset/map/field/mapper lineage differs")
    contributors = _load_contributors(Path(args.contributors))
    device = torch.device(str(args.device))
    mapper, _ = load_surface_maplet_mapper(Path(args.surface_mapper), device=str(device))
    mapper.model.to(device).eval()
    scene = FrozenSoftSurfaceSceneGPU(physical, field, device=str(device))
    begin = int(args.query_start)
    end = int(arrays["image_ids"].size)
    if args.maximum_queries is not None:
        end = min(end, begin + int(args.maximum_queries))
    if not 0 <= begin < end:
        raise ValueError("query range is empty")

    feature_output = Path(args.compact_feature_output) if args.compact_feature_output else None
    feature_sidecar = None if feature_output is None else feature_output.with_suffix(".json")
    feature_partial = None if feature_output is None else feature_output.with_suffix(".partial.npy")
    compact_features = None
    if feature_output is not None:
        if feature_output.suffix != ".npy":
            raise ValueError("compact feature output must use .npy suffix")
        if any(path.exists() for path in (feature_output, feature_sidecar, feature_partial)):
            raise FileExistsError("refusing to overwrite compact full-token features")
        feature_output.parent.mkdir(parents=True, exist_ok=True)
        compact_features = np.lib.format.open_memmap(
            feature_partial,
            mode="w+",
            dtype=np.float16,
            shape=(
                end - begin,
                int(arrays["candidate_valid"].shape[1]),
                len(FULLTOKEN_POSE_RANKING_CHANNELS),
                36,
                64,
            ),
        )
        compact_features[:] = 0.0

    score_rows = np.full(arrays["candidate_valid"].shape, -1.0, dtype=np.float32)
    component_rows: list[list[dict[str, object] | None]] = [
        [None] * int(arrays["candidate_valid"].shape[1])
        for _ in range(int(arrays["image_ids"].size))
    ]
    timing_rows = []
    with torch.no_grad():
        for query_index in range(begin, end):
            image_id = str(arrays["image_ids"][query_index])
            if image_id not in contributors:
                raise ValueError(f"missing contributor camera for {image_id}")
            contributor_path = contributors[image_id]
            if "contributor_paths" in arrays:
                declared = Path(str(arrays["contributor_paths"][query_index])).resolve()
                if declared != contributor_path or file_sha256(declared) != str(
                    arrays["contributor_file_sha256"][query_index]
                ):
                    raise ValueError("direct candidate contributor lineage differs")
            camera = _camera(contributor_path)
            if "radio_final" in arrays:
                raw_np = np.asarray(arrays["radio_final"][query_index], dtype=np.float32)
            else:
                token_path = Path(str(arrays["radio_token_paths"][query_index])).resolve()
                if file_sha256(token_path) != str(arrays["radio_file_sha256"][query_index]):
                    raise ValueError("direct candidate RADIO token lineage differs")
                with np.load(token_path, allow_pickle=False) as data:
                    if set(data.files) != {"radio_final"}:
                        raise ValueError("RADIO token NPZ members differ")
                    raw_np = np.asarray(data["radio_final"], dtype=np.float32)
            if raw_np.shape != (1280, 36, 64) or np.any(~np.isfinite(raw_np)):
                raise ValueError("direct candidate RADIO tensor differs")
            raw = torch.as_tensor(raw_np, device=device, dtype=torch.float32)[None]
            query = mapper.model(raw)[0].permute(1, 2, 0).reshape(2304, 128)
            valid_rows = np.flatnonzero(arrays["candidate_valid"][query_index])
            query_timing = {
                "raster_seconds": 0.0,
                "token_remap_seconds": 0.0,
                "direct_reduction_seconds": 0.0,
                "total_render_seconds": 0.0,
                "packed_hit_count": 0,
                "remapped_hit_count": 0,
            }
            for start in range(0, int(valid_rows.size), int(args.render_batch_size)):
                rows = valid_rows[start:start + int(args.render_batch_size)]
                rendered = scene.render_direct_canonical_grid_batch(
                    arrays["candidate_poses_w2c"][query_index, rows],
                    camera,
                )
                for key in (
                    "raster_seconds", "token_remap_seconds", "direct_reduction_seconds"
                ):
                    query_timing[key] += float(getattr(rendered, key))
                query_timing["total_render_seconds"] += float(rendered.total_seconds)
                query_timing["packed_hit_count"] += int(rendered.packed_hit_count)
                query_timing["remapped_hit_count"] += int(rendered.remapped_hit_count)
                for local, candidate in enumerate(rows.tolist()):
                    target = torch.as_tensor(
                        rendered.feature[local].reshape(2304, 1, -1),
                        device=device,
                        dtype=torch.float32,
                    )
                    mass = torch.as_tensor(
                        rendered.mass[local].reshape(2304, 1),
                        device=device,
                        dtype=torch.float32,
                    )
                    valid = torch.as_tensor(
                        rendered.valid[local].reshape(2304, 1), device=device
                    )
                    if str(args.score_semantics) == "conditional_phase_shift_control":
                        result = conditional_fulltoken_phase_pose_energy_control(
                            query,
                            target,
                            mass,
                            valid,
                            height=36,
                            width=64,
                            maximum_shift_tokens=int(args.phase_shift_radius_tokens),
                        )
                    else:
                        result = conservative_fulltoken_phase_pose_energy(
                            query,
                            target,
                            mass,
                            valid,
                            height=36,
                            width=64,
                            maximum_shift_tokens=int(args.phase_shift_radius_tokens),
                            slot_pair_reduction=(
                                "maximum_bottleneck"
                                if str(args.score_semantics) == "conservative_phase_shift_maxmin"
                                else "additive_product"
                            ),
                        )
                    score_rows[query_index, candidate] = float(result.score.cpu())
                    if compact_features is not None:
                        compact_features[query_index - begin, candidate] = (
                            compact_fulltoken_pose_ranking_features(
                                query,
                                target,
                                mass,
                                valid,
                                height=36,
                                width=64,
                            ).cpu().numpy().astype(np.float16, copy=False)
                        )
                    content = result.conditional_content_score
                    information = result.mass_observability
                    if content is not None and information is not None:
                        component_rows[query_index][candidate] = {
                            "conditional_content_score": float(content.cpu()),
                            "mass_observability": float(information.cpu()),
                            "selected_shift_y": int(result.selected_shift_y),
                            "selected_shift_x": int(result.selected_shift_x),
                        }
            timing_rows.append({"image_id": image_id, **query_timing})

    # Candidate scores and rendering timings are frozen before labels are
    # consumed by the following metrics.
    train_count = min(max(int(args.train_queries), 0), end - begin)
    scores = score_rows[begin:end]
    valid = arrays["candidate_valid"][begin:end]
    energy_semantics = (
        CONDITIONAL_SHIFT_PHASE_CONTROL_SEMANTICS
        if str(args.score_semantics) == "conditional_phase_shift_control"
        else (
            SHIFT_TOLERANT_MAXMIN_PHASE_SEMANTICS
            if str(args.score_semantics) == "conservative_phase_shift_maxmin"
            else SHIFT_TOLERANT_PHASE_SEMANTICS
        )
    )
    report = {
        "artifact_type": SCHEMA,
        "score_semantics": str(args.score_semantics),
        "energy_semantics": energy_semantics,
        "render_semantics": "full_scene_occluded_direct_canonical_token_feature_grid_v1",
        "dataset_file_sha256": file_sha256(dataset_path),
        "dataset_content_sha256": metadata["content_sha256"],
        "physical_map_sha256": physical.content_sha256,
        "canonical_field_sha256": field.content_sha256,
        "surface_mapper_file_sha256": mapper_sha,
        "phase_shift_radius_tokens": int(args.phase_shift_radius_tokens),
        "query_range": [begin, end],
        "candidate_score": scores.tolist(),
        "candidate_phase_evidence_components": component_rows[begin:end],
        "timing_rows": timing_rows,
        "mean_render_seconds_per_query": float(np.mean([
            row["total_render_seconds"] for row in timing_rows
        ])),
        "all_candidate_scores_built_before_pose_error_metrics": True,
        "all_metrics": _metrics(
            scores,
            arrays["translation_m"][begin:end],
            arrays["rotation_deg"][begin:end],
            valid,
            arrays["image_ids"][begin:end],
        ),
        "all_distinct_basin_recall": distinct_pose_basin_recall_metrics(
            scores,
            arrays["candidate_poses_w2c"][begin:end],
            arrays["translation_m"][begin:end],
            arrays["rotation_deg"][begin:end],
            valid,
        ),
        "all_continuous_domain_acquisition_8m_45deg": (
            continuous_seed_domain_recall_metrics(
                scores,
                arrays["candidate_poses_w2c"][begin:end],
                valid,
                translation_half_extent_m=8.0,
                rotation_radius_deg=45.0,
            )
        ),
        "prefix_metrics": _metrics(
            scores[:train_count],
            arrays["translation_m"][begin:begin + train_count],
            arrays["rotation_deg"][begin:begin + train_count],
            valid[:train_count],
            arrays["image_ids"][begin:begin + train_count],
        ) if train_count else None,
        "suffix_metrics": _metrics(
            scores[train_count:],
            arrays["translation_m"][begin + train_count:end],
            arrays["rotation_deg"][begin + train_count:end],
            valid[train_count:],
            arrays["image_ids"][begin + train_count:end],
        ) if train_count < end - begin else None,
        "uses_stored_target_child_top_l": False,
        "uses_alike": False,
        "uses_point_correspondences": False,
        "uses_pnp": False,
        "uses_absolute_pose_regression": False,
        "production_eligible": False,
        "claim": "candidate_conditioned_pose_energy_diagnostic_not_localization_success",
    }
    if compact_features is not None:
        compact_features.flush()
        del compact_features
        os.replace(feature_partial, feature_output)
        feature_manifest = {
            "artifact_type": "goal_maplet_compact_fulltoken_pose_ranking_features_v1",
            "feature_semantics": FULLTOKEN_POSE_RANKING_FEATURE_SEMANTICS,
            "feature_channels": list(FULLTOKEN_POSE_RANKING_CHANNELS),
            "feature_shape": [
                end - begin,
                int(arrays["candidate_valid"].shape[1]),
                len(FULLTOKEN_POSE_RANKING_CHANNELS),
                36,
                64,
            ],
            "feature_dtype": "float16",
            "feature_file": str(feature_output.resolve()),
            "feature_file_sha256": file_sha256(feature_output),
            "dataset_file_sha256": file_sha256(dataset_path),
            "dataset_content_sha256": metadata["content_sha256"],
            "query_range": [begin, end],
            "candidate_scores_and_features_built_before_pose_error_metrics": True,
            "signed_channels_are_empirical_and_not_missingness_monotone": True,
            "above_floor_channels_are_nonnegative_fixed_atom_evidence": True,
            "uses_alike": False,
            "uses_point_correspondences": False,
            "uses_pnp": False,
            "uses_absolute_pose_regression": False,
            "production_eligible": False,
        }
        feature_sidecar.write_text(json.dumps(feature_manifest, indent=2, sort_keys=True) + "\n")
        report["compact_feature_manifest"] = str(feature_sidecar.resolve())
        report["compact_feature_file_sha256"] = feature_manifest["feature_file_sha256"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output),
        "mean_render_seconds_per_query": report["mean_render_seconds_per_query"],
        "metrics": {key: value for key, value in report["all_metrics"].items() if key != "rows"},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
