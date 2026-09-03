"""Rank finite map planes directly from query-region RADIO descriptors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import _radio, _records, _region_token_support
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256, canonical_json_sha256, file_sha256
from feature_extract.vfm.localization_goal_maplet.query_plane_regions import QueryPlaneRegions


def _base_to_carrier_regions(base_labels: np.ndarray, carrier_labels: np.ndarray) -> np.ndarray:
    """Map every connected base region to exactly one observed-only carrier."""

    base = np.asarray(base_labels, np.int32)
    carrier = np.asarray(carrier_labels, np.int32)
    if base.shape != carrier.shape or not np.array_equal(base >= 0, carrier >= 0):
        raise ValueError("base/carrier observed support differs")
    output = np.full(int(base.max()) + 1 if np.any(base >= 0) else 0, -1, np.int32)
    for region in range(len(output)):
        values = np.unique(carrier[base == region])
        if len(values) != 1 or int(values[0]) < 0:
            raise ValueError("a connected base region maps to multiple carriers")
        output[region] = int(values[0])
    return output


def _aggregate_observation_scores(
    observation_score: np.ndarray,
    plane_offsets: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Reduce per-view scores without rewarding a single accidental peak."""

    score = np.asarray(observation_score, np.float64)
    offsets = np.asarray(plane_offsets, np.int64)
    if mode == "max":
        return np.maximum.reduceat(score, offsets[:-1])
    if mode != "top2_mean":
        raise ValueError("unsupported plane-observation aggregation")
    output = np.empty(len(offsets) - 1, np.float64)
    for plane, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:])):
        values = score[int(lo):int(hi)]
        if not len(values):
            raise ValueError("plane has no source observation")
        output[plane] = (
            float(values[0]) if len(values) == 1
            else float(np.mean(np.partition(values, -2)[-2:]))
        )
    return output


def _query_subset_names(path: Path | None) -> list[str] | None:
    """Load a frozen, pose-free query order used only to shard scoring work."""

    if path is None:
        return None
    payload = json.loads(path.read_text())
    if (
        payload.get("artifact_type")
        not in {
            "goal_maplet_direct_radio_to_finite_plane_ranking_v1",
            "goal_maplet_direct_radio_to_finite_plane_ranking_v2",
            "goal_maplet_context_fused_direct_radio_to_finite_plane_ranking_v1",
            "goal_maplet_context_fused_direct_radio_to_finite_plane_ranking_v2",
        }
        or payload.get("uses_pose_or_ground_truth") is not False
        or payload.get("contains_postlabel_fields") is not False
    ):
        raise ValueError("query subset inventory is not pose/label-free")
    names = [str(row["image"]) for row in payload.get("rows", [])]
    if not names or len(set(names)) != len(names):
        raise ValueError("query subset inventory is empty or duplicated")
    return names


def _consensus_supplement_ranking(
    own_plane_score: np.ndarray,
    group_plane_scores: np.ndarray,
    *,
    topk: int,
    preserve: int,
    vote_depth: int,
    minimum_votes: int,
) -> tuple[np.ndarray, int, int]:
    """Preserve an island ranking and append one independently voted plane."""

    own = np.asarray(own_plane_score, np.float64)
    group = np.asarray(group_plane_scores, np.float64)
    if own.ndim != 1 or group.ndim != 2 or group.shape[1:] != own.shape:
        raise ValueError("candidate consensus score shapes differ")
    if not (1 <= preserve <= topk <= len(own)) or vote_depth < 1 or minimum_votes < 2:
        raise ValueError("candidate consensus budgets differ")
    plane_ids = np.arange(len(own), dtype=np.int64)
    own_ranking = np.lexsort((plane_ids, -own))
    base = own_ranking[:preserve].tolist()
    votes = np.zeros(len(own), np.int32)
    score_sum = np.zeros(len(own), np.float64)
    for scores in group:
        ranking = np.lexsort((plane_ids, -scores))[:vote_depth]
        votes[ranking] += 1
        score_sum[ranking] += scores[ranking]
    eligible = np.flatnonzero(votes >= int(minimum_votes))
    consensus = sorted(
        eligible.tolist(),
        key=lambda plane: (
            -int(votes[plane]),
            -float(score_sum[plane] / max(int(votes[plane]), 1)),
            int(plane),
        ),
    )
    supplement = next((int(plane) for plane in consensus if int(plane) not in base), -1)
    output = list(base)
    if supplement >= 0:
        output.append(supplement)
    output.extend(int(plane) for plane in own_ranking.tolist() if int(plane) not in output)
    return (
        np.asarray(output[:topk], np.int64),
        supplement,
        0 if supplement < 0 else int(votes[supplement]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plane_field", type=Path, required=True)
    parser.add_argument("--query_plane_dir", type=Path, required=True)
    parser.add_argument("--sparse_occlusion_carrier_dir", type=Path)
    parser.add_argument(
        "--candidate_sharing_mode",
        choices=("max_replace", "preserve_then_supplement", "consensus_supplement"),
        default="max_replace",
    )
    parser.add_argument("--candidate_sharing_preserve", type=int, default=5)
    parser.add_argument("--candidate_sharing_vote_depth", type=int, default=10)
    parser.add_argument("--candidate_sharing_minimum_votes", type=int, default=2)
    parser.add_argument("--radio_manifest", type=Path, required=True)
    parser.add_argument(
        "--query_subset_inventory", type=Path,
        help="Pose-free ranking whose exact query order shards this ranking run.",
    )
    parser.add_argument("--context_field", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument(
        "--query_region_descriptor_weighting",
        choices=("uniform", "visible_pixel_fraction"), default="uniform",
    )
    parser.add_argument(
        "--plane_observation_aggregation", choices=("max", "top2_mean"), default="max",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite direct plane RADIO ranking")
    with np.load(args.plane_field, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        arrays = {name: np.asarray(data[name]) for name in (
            "plane_offsets", "observation_plane_rows", "observation_descriptors",
        )}
    if (
        metadata.get("artifact_type") != "goal_maplet_direct_radio_plane_observation_field_v1"
        or arrays_sha256(arrays) != metadata.get("arrays_sha256")
    ):
        raise ValueError("direct plane RADIO field contract differs")
    offsets = arrays["plane_offsets"]
    descriptor = np.asarray(arrays["observation_descriptors"], np.float32)
    token_grid = tuple(map(int, metadata.get("token_grid", (36, 64))))
    if np.any(np.diff(offsets) <= 0):
        raise ValueError("every plane must have a RADIO observation")
    context_descriptor = context_meta = None
    if args.context_field is not None:
        with np.load(args.context_field, allow_pickle=False) as data:
            context_meta = json.loads(str(data["metadata_json"].item()))
            context_arrays = {name: np.asarray(data[name]) for name in (
                "plane_offsets", "observation_context_descriptors",
            )}
        if (
            context_meta.get("artifact_type") != "goal_maplet_plane_observation_context_field_v1"
            or context_meta.get("plane_field_content_sha256") != metadata.get("content_sha256")
            or arrays_sha256(context_arrays) != context_meta.get("arrays_sha256")
            or not np.array_equal(context_arrays["plane_offsets"], offsets)
            or context_arrays["observation_context_descriptors"].shape != descriptor.shape
        ):
            raise ValueError("plane context field contract differs")
        context_descriptor = np.asarray(context_arrays["observation_context_descriptors"], np.float32)
    records = _records([args.radio_manifest])
    query_manifest_path = args.query_plane_dir / "manifest.json"
    query_manifest = json.loads(query_manifest_path.read_text())
    if (
        query_manifest.get("artifact_type") not in (
            "goal_maplet_query_plane_region_cache_run_v2",
            "goal_maplet_query_plane_region_cache_run_v3",
        )
        or query_manifest.get("uses_pose_or_ground_truth") is not False
    ):
        raise ValueError("query-plane run manifest contract differs")
    query_paths = sorted(args.query_plane_dir.glob("*.npz"))
    if [path.name for path in query_paths] != list(query_manifest.get("selected_names_in_order", [])):
        raise ValueError("query-plane inventory differs from its run manifest")
    subset_names = _query_subset_names(args.query_subset_inventory)
    if subset_names is not None:
        path_by_name = {path.name: path for path in query_paths}
        if any(name not in path_by_name for name in subset_names):
            raise ValueError("query subset is not contained in the query-plane inventory")
        query_paths = [path_by_name[name] for name in subset_names]
    sparse_carrier = bool(query_manifest.get("sparse_occlusion_carrier", False))
    if sparse_carrier and (
        query_manifest.get("hidden_pixel_count_added") != 0
        or query_manifest.get("observed_support_only") is not True
    ):
        raise ValueError("sparse-occlusion carrier is allowed to group observed pixels only")
    candidate_sharing = args.sparse_occlusion_carrier_dir is not None
    if not candidate_sharing and args.candidate_sharing_mode != "max_replace":
        raise ValueError("candidate-sharing mode requires a carrier directory")
    if int(args.candidate_sharing_preserve) < 1 or int(args.candidate_sharing_preserve) > int(args.topk):
        raise ValueError("candidate-sharing preserve budget is outside topk")
    if int(args.candidate_sharing_vote_depth) < 1 or int(args.candidate_sharing_minimum_votes) < 2:
        raise ValueError("candidate-sharing vote contract differs")
    carrier_manifest_path = None
    carrier_manifest = None
    carrier_inventory = []
    if candidate_sharing:
        if sparse_carrier:
            raise ValueError("candidate sharing requires the connected-region query inventory")
        carrier_manifest_path = args.sparse_occlusion_carrier_dir / "manifest.json"
        carrier_manifest = json.loads(carrier_manifest_path.read_text())
        if (
            carrier_manifest.get("artifact_type") != "goal_maplet_query_plane_region_cache_run_v3"
            or carrier_manifest.get("uses_pose_or_ground_truth") is not False
            or carrier_manifest.get("sparse_occlusion_carrier") is not True
            or carrier_manifest.get("observed_support_only") is not True
            or carrier_manifest.get("hidden_pixel_count_added") != 0
            or list(carrier_manifest.get("selected_names_in_order", []))
            != [path.name for path in query_paths]
        ):
            raise ValueError("candidate-sharing carrier manifest differs")
    rows = []
    query_plane_inventory = []
    for path in query_paths:
        planes, plane_meta = QueryPlaneRegions.load_npz(path)
        if bool(plane_meta.get("sparse_occlusion_carrier", False)) != sparse_carrier:
            raise ValueError("query-plane member carrier semantics differ from its manifest")
        query_plane_inventory.append({
            "name": path.name,
            "file_sha256": file_sha256(path),
            "content_sha256": plane_meta.get("content_sha256"),
        })
        carrier_planes = carrier_meta = None
        base_to_carrier = None
        if candidate_sharing:
            carrier_path = args.sparse_occlusion_carrier_dir / path.name
            carrier_planes, carrier_meta = QueryPlaneRegions.load_npz(carrier_path)
            carrier_diagnostics = carrier_meta.get("carrier_diagnostics", {})
            if (
                carrier_meta.get("sparse_occlusion_carrier") is not True
                or carrier_diagnostics.get("observed_support_bit_exact") is not True
                or carrier_diagnostics.get("hidden_pixel_count_added") != 0
            ):
                raise ValueError("candidate-sharing carrier member differs")
            base_to_carrier = _base_to_carrier_regions(planes.labels, carrier_planes.labels)
            carrier_inventory.append({
                "name": carrier_path.name,
                "file_sha256": file_sha256(carrier_path),
                "content_sha256": carrier_meta.get("content_sha256"),
            })
        query = _radio(path.name, records)
        if query.shape[0] != int(np.prod(token_grid)):
            raise ValueError("query RADIO grid differs from plane field")
        region_payload = []
        context_sum = np.zeros(1280, np.float64)
        context_weight = 0
        for region in range(len(planes.normals_camera)):
            tokens, weights = _region_token_support(
                planes.labels, region, token_grid=token_grid,
            )
            qdescriptor = (
                np.average(query[tokens], axis=0, weights=weights)
                if len(tokens) and args.query_region_descriptor_weighting == "visible_pixel_fraction"
                else np.mean(query[tokens], axis=0) if len(tokens)
                else np.zeros(1280, np.float32)
            )
            qdescriptor /= max(float(np.linalg.norm(qdescriptor)), 1e-8)
            region_payload.append((region, tokens, qdescriptor))
            region_weight = float(np.sum(weights)) if args.query_region_descriptor_weighting == "visible_pixel_fraction" else len(tokens)
            context_sum += np.asarray(qdescriptor, np.float64) * region_weight
            context_weight += region_weight
        qcontext = context_sum / max(context_weight, 1)
        qcontext /= max(float(np.linalg.norm(qcontext)), 1e-8)
        region_scores = []
        for region, tokens, qdescriptor in region_payload:
            observation_score = descriptor @ qdescriptor
            if context_descriptor is not None:
                context_score = context_descriptor @ qcontext.astype(np.float32)
                # Fixed, zero-parameter geometric mean in [0, 1].  The affine
                # cosine mapping is monotone for either signal individually.
                observation_score = np.sqrt(
                    np.clip((observation_score + 1.0) * 0.5, 0.0, 1.0)
                    * np.clip((context_score + 1.0) * 0.5, 0.0, 1.0)
                )
            plane_score = _aggregate_observation_scores(
                observation_score, offsets, str(args.plane_observation_aggregation),
            )
            region_scores.append(plane_score)
        shared_scores: dict[int, np.ndarray] = {}
        if candidate_sharing:
            assert base_to_carrier is not None
            for carrier_region in np.unique(base_to_carrier).tolist():
                members = np.flatnonzero(base_to_carrier == int(carrier_region))
                shared_scores[int(carrier_region)] = np.max(
                    np.stack([region_scores[int(member)] for member in members], axis=0), axis=0,
                )
        regions = []
        for (region, tokens, _), own_plane_score in zip(region_payload, region_scores):
            carrier_region = int(base_to_carrier[region]) if candidate_sharing else int(region)
            plane_score = shared_scores[carrier_region] if candidate_sharing else own_plane_score
            supplement = -1
            supplement_votes = 0
            if candidate_sharing and args.candidate_sharing_mode == "consensus_supplement":
                members = np.flatnonzero(base_to_carrier == carrier_region)
                ranking, supplement, supplement_votes = _consensus_supplement_ranking(
                    own_plane_score,
                    np.stack([region_scores[int(member)] for member in members], axis=0),
                    topk=int(args.topk),
                    preserve=int(args.candidate_sharing_preserve),
                    vote_depth=int(args.candidate_sharing_vote_depth),
                    minimum_votes=int(args.candidate_sharing_minimum_votes),
                )
            elif candidate_sharing and args.candidate_sharing_mode == "preserve_then_supplement":
                own_ranking = np.lexsort((np.arange(len(own_plane_score)), -own_plane_score))
                shared_ranking = np.lexsort((np.arange(len(plane_score)), -plane_score))
                ranking_list = own_ranking[: int(args.candidate_sharing_preserve)].tolist()
                ranking_list.extend(
                    int(value) for value in shared_ranking.tolist()
                    if int(value) not in ranking_list
                )
                ranking_list.extend(
                    int(value) for value in own_ranking.tolist()
                    if int(value) not in ranking_list
                )
                ranking = np.asarray(ranking_list[: int(args.topk)], np.int64)
            else:
                ranking = np.lexsort((np.arange(len(plane_score)), -plane_score))[: int(args.topk)]
            regions.append({
                "region": int(region),
                "carrier_region": carrier_region,
                "carrier_consensus_supplement_plane": int(supplement),
                "carrier_consensus_supplement_votes": int(supplement_votes),
                "pixels": int(np.sum(planes.labels == region)),
                "top10": ranking.tolist(),
                "top10_scores": plane_score[ranking].tolist(),
            })
        rows.append({"image": path.name, "query_plane_count": len(regions), "regions": regions})
    if not rows:
        raise ValueError("query plane directory contains no NPZ inputs")
    report = {
        "artifact_type": (
            "goal_maplet_context_fused_direct_radio_to_finite_plane_ranking_v2"
            if context_descriptor is not None
            else "goal_maplet_direct_radio_to_finite_plane_ranking_v2"
        ),
        "query_count": len(rows),
        "plane_count": int(len(offsets) - 1),
        "uses_pose_or_ground_truth": False,
        "contains_postlabel_fields": False,
        "query_plane_manifest_file_sha256": file_sha256(query_manifest_path),
        "query_plane_inventory_sha256": canonical_json_sha256(query_plane_inventory),
        "query_plane_inventory": query_plane_inventory,
        "sparse_occlusion_carrier": sparse_carrier,
        "sparse_occlusion_carrier_config": query_manifest.get("carrier_config"),
        "hidden_pixel_count_added": int(query_manifest.get("hidden_pixel_count_added", 0)),
        "sparse_occlusion_candidate_sharing": candidate_sharing,
        "candidate_sharing_semantics": (
            None if not candidate_sharing else
            "preserve_connected_region_candidates_then_append_multi_island_vote;_"
            "minimum_two_independent_components;_component_matching_remains_independent"
            if args.candidate_sharing_mode == "consensus_supplement" else
            "preserve_connected_region_candidates_then_append_unique_group_candidates;_"
            "component_RADIO_and_homography_matching_remain_independent"
            if args.candidate_sharing_mode == "preserve_then_supplement" else
            "max_plane_retrieval_score_across_observed_connected_components;_"
            "component_RADIO_and_homography_matching_remain_independent"
        ),
        "candidate_sharing_mode": (
            None if not candidate_sharing else str(args.candidate_sharing_mode)
        ),
        "candidate_sharing_preserve_count": (
            None if not candidate_sharing else int(args.candidate_sharing_preserve)
        ),
        "candidate_sharing_vote_depth": (
            None if not candidate_sharing else int(args.candidate_sharing_vote_depth)
        ),
        "candidate_sharing_minimum_votes": (
            None if not candidate_sharing else int(args.candidate_sharing_minimum_votes)
        ),
        "candidate_sharing_carrier_manifest_file_sha256": (
            None if carrier_manifest_path is None else file_sha256(carrier_manifest_path)
        ),
        "candidate_sharing_carrier_inventory_sha256": (
            None if not candidate_sharing else canonical_json_sha256(carrier_inventory)
        ),
        "candidate_sharing_carrier_inventory": carrier_inventory,
        "candidate_sharing_carrier_config": (
            None if carrier_manifest is None else carrier_manifest.get("carrier_config")
        ),
        "query_region_descriptor_weighting": str(args.query_region_descriptor_weighting),
        "plane_observation_aggregation": str(args.plane_observation_aggregation),
        "plane_field_file_sha256": file_sha256(args.plane_field),
        "plane_field_content_sha256": metadata.get("content_sha256"),
        "context_field_file_sha256": None if args.context_field is None else file_sha256(args.context_field),
        "context_field_content_sha256": None if context_meta is None else context_meta.get("content_sha256"),
        "fusion": None if context_descriptor is None else "geometric_mean_of_affine_local_and_view_context_cosine",
        "query_radio_manifest_file_sha256": file_sha256(args.radio_manifest),
        "query_subset_inventory_file_sha256": (
            None if args.query_subset_inventory is None
            else file_sha256(args.query_subset_inventory)
        ),
        "topk": int(args.topk),
        "token_grid": list(token_grid),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
