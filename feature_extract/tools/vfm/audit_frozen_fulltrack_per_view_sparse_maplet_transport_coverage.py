"""Audit the target-free coverage of paired v2 sparse-SfM-maplet evidence.

The sparse-maplet v1 probe accidentally required populated support neighbours
in all four quadrants.  This audit verifies that v2's partial-maplet rule is
actually retained across every frozen global-top20 rank bucket.  It also binds
each visual family to an exactly matched descriptor-free topology control:
the two sides must have identical frozen CSR rows, geometry edges, support
neighbour counts, and selected-edge validity before an identity OOF probe is
allowed to compare them.

No identity labels, residuals, query poses, hypotheses, scores, retrieval, or
rendered data are loaded here.  Only CSR headers, edge-validity masks, and the
explicit topology-control neighbour counts are streamed one shard at a time.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.tools.vfm.audit_frozen_fulltrack_hard_pairs import (
    _validate_projected_bank_lineage,
)
from feature_extract.tools.vfm.audit_frozen_fulltrack_per_view_region_context_coverage import (
    EXPECTED_QUERY_COUNTS,
    RANK_BUCKETS,
)
from feature_extract.tools.vfm.audit_frozen_fulltrack_per_view_spatial_pyramid_shift_coverage import (
    _StreamingArtifactHeader,
    _frozen_candidate_ranks,
    _load_streaming_headers,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.frozen_fulltrack_per_view_candidate_probe import (
    FULLTRACK_PER_VIEW_FAMILIES,
    FULLTRACK_PER_VIEW_SPARSE_MAPLET_PAIRED_CONTROL_FAMILIES,
    FULLTRACK_PER_VIEW_SPARSE_MAPLET_TOPOLOGY_CONTROL_EDGE_FEATURE_SEMANTICS,
    FULLTRACK_PER_VIEW_SPARSE_MAPLET_TRANSPORT_EDGE_FEATURE_SEMANTICS,
    profile_indices_for_fulltrack_per_view_family,
)
from feature_extract.vfm.localization.frozen_fulltrack_sparse_maplet_transport import (
    FULLTRACK_SPARSE_MAPLET_TOPOLOGY_CONTROL_APPEARANCE_FORMAT,
    FULLTRACK_SPARSE_MAPLET_TRANSPORT_APPEARANCE_FORMAT,
    SPARSE_MAPLET_MAX_NEIGHBORS_PER_QUADRANT,
    SPARSE_MAPLET_MINIMUM_TOTAL_NEIGHBORS,
    SPARSE_MAPLET_TOPOLOGY_CONTROL_FEATURE_NAMES,
    SPARSE_MAPLET_TOPOLOGY_CONTROL_FEATURE_NAMES_BY_PROFILE,
    SPARSE_MAPLET_TRANSPORT_FEATURE_NAMES,
    SPARSE_MAPLET_TRANSPORT_FEATURE_NAMES_BY_PROFILE,
    SPARSE_MAPLET_TRANSPORT_PROFILES,
)


ARTIFACT_FORMAT = "frozen_fulltrack_per_view_sparse_maplet_transport_coverage_audit_v2"
EXPORT_MANIFEST_FORMAT = "frozen_fulltrack_sparse_maplet_transport_export_v2"
_VISUAL_ROLE = "visual_descriptor_transport"
_CONTROL_ROLE = "topology_and_original_crop_control"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-manifest", required=True)
    parser.add_argument("--projected-landmark-bank", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--visual-families",
        default="all",
        help="comma-separated paired sparse-maplet visual families; all selects the frozen v2 sweep",
    )
    return parser.parse_args(argv)


def _visual_families(value: str) -> tuple[str, ...]:
    if str(value).strip() == "all":
        names = tuple(FULLTRACK_PER_VIEW_SPARSE_MAPLET_PAIRED_CONTROL_FAMILIES)
    else:
        names = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if (
        not names
        or len(set(names)) != len(names)
        or set(names).difference(FULLTRACK_PER_VIEW_SPARSE_MAPLET_PAIRED_CONTROL_FAMILIES)
        or any(
            FULLTRACK_PER_VIEW_FAMILIES[name].edge_feature_semantics
            != FULLTRACK_PER_VIEW_SPARSE_MAPLET_TRANSPORT_EDGE_FEATURE_SEMANTICS
            for name in names
        )
    ):
        raise ValueError("sparse-maplet visual families are invalid or unpaired")
    return names


def _load_completed_export_paths(export_manifest: Path) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Resolve a complete exporter manifest and recheck every frozen output hash."""

    manifest_path = Path(export_manifest)
    payload = json.loads(manifest_path.read_text())
    if (
        not isinstance(payload, Mapping)
        or payload.get("format") != EXPORT_MANIFEST_FORMAT
        or payload.get("complete") is not True
        or not isinstance(payload.get("config"), Mapping)
        or not isinstance(payload.get("completed"), Mapping)
        or not isinstance(payload.get("failed"), Mapping)
        or payload["failed"]
    ):
        raise ValueError("sparse-maplet export manifest is incomplete or incompatible")
    sources = payload["config"].get("source_manifest")
    completed = payload["completed"]
    if (
        not isinstance(sources, list)
        or not sources
        or len(completed) != len(sources)
    ):
        raise ValueError("sparse-maplet export manifest has incomplete source coverage")
    visual: list[Path] = []
    control: list[Path] = []
    seen: set[str] = set()
    for item in sources:
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
            raise ValueError("sparse-maplet export source manifest is invalid")
        source = Path(str(item["path"])).resolve()
        key = source.parent.name
        entry = completed.get(key)
        if (
            key in seen
            or not isinstance(entry, Mapping)
            or Path(str(entry.get("source", ""))).resolve() != source
        ):
            raise ValueError("sparse-maplet exporter shard/source pairing is invalid")
        seen.add(key)
        for path_key, hash_key, destination in (
            ("visual_output", "visual_output_sha256", visual),
            ("control_output", "control_output_sha256", control),
        ):
            output = Path(str(entry.get(path_key, ""))).resolve()
            expected_hash = str(entry.get(hash_key, ""))
            if not output.is_file() or not expected_hash:
                raise ValueError("sparse-maplet exporter output is missing")
            if file_sha256_short(output) != expected_hash:
                raise ValueError("sparse-maplet exporter output hash is stale")
            destination.append(output)
    if len(visual) != len(control) or len(set(visual)) != len(visual):
        raise ValueError("sparse-maplet exporter output paths are duplicated")
    return tuple(visual), tuple(control)


def _validate_sparse_maplet_metadata(
    *,
    paths: Sequence[Path],
    headers: Sequence[_StreamingArtifactHeader],
    control: bool,
) -> None:
    expected_names = (
        SPARSE_MAPLET_TOPOLOGY_CONTROL_FEATURE_NAMES
        if control
        else SPARSE_MAPLET_TRANSPORT_FEATURE_NAMES
    )
    expected_format = (
        FULLTRACK_SPARSE_MAPLET_TOPOLOGY_CONTROL_APPEARANCE_FORMAT
        if control
        else FULLTRACK_SPARSE_MAPLET_TRANSPORT_APPEARANCE_FORMAT
    )
    expected_role = _CONTROL_ROLE if control else _VISUAL_ROLE
    if len(paths) != len(headers) or not headers:
        raise ValueError("sparse-maplet metadata rows are incomplete")
    if tuple(headers[0].profile_names) != tuple(expected_names):
        raise ValueError("sparse-maplet feature order differs from its frozen contract")
    expected_profile_contract = [
        {
            "name": profile.name,
            "source": profile.source_name,
            "grid_size": int(profile.grid_size),
            "radius_cells": int(profile.radius_cells),
            "window_size": int(profile.window_size),
        }
        for profile in SPARSE_MAPLET_TRANSPORT_PROFILES
    ]
    for path, header in zip(paths, headers):
        metadata = header.metadata
        strict = metadata.get("strict_fulltrack_appearance_contract")
        appearance = metadata.get("appearance_config")
        contract = metadata.get("sparse_maplet_transport_contract")
        profiles = contract.get("profiles") if isinstance(contract, Mapping) else None
        if (
            metadata.get("format") != expected_format
            or metadata.get("artifact_role") != expected_role
            or metadata.get("contains_ground_truth") is not False
            or metadata.get("contains_target_errors") is not False
            or metadata.get("pose_or_ground_truth_used") is not False
            or metadata.get("supervision_arrays_loaded") is not False
            or metadata.get("fixed_candidate_top_k") != 20
            or metadata.get("support_view_source")
            != "all_real_sfm_track_observations_v1"
            or metadata.get("support_view_count_cap") is not None
            or metadata.get("per_view_edges_retained") is not True
            or not isinstance(strict, Mapping)
            or strict.get("candidate_identity_fixed") is not True
            or strict.get("candidate_posterior_preserved") is not True
            or strict.get("candidate_reselection") is not False
            or strict.get("support_reselection") is not False
            or strict.get("all_real_sfm_track_observations_enumerated") is not True
            or strict.get("support_view_count_cap") is not None
            or strict.get("candidate_3d_projection_or_pose_used") is not False
            or strict.get("image_retrieval_or_submap_used") is not False
            or strict.get("render") is not False
            or strict.get("candidate_center_descriptor_excluded") is not True
            or strict.get("partial_support_maplet_is_retained_when_total_neighbors_sufficient")
            is not True
            or strict.get("visual_descriptor_values_included") is not (not control)
            or strict.get("topology_or_original_crop_values_included") is not control
            or strict.get("paired_topology_control_artifact_required") is not True
            or not isinstance(appearance, Mapping)
            or appearance.get("candidate_specific") is not True
            or appearance.get("per_view") is not True
            or appearance.get("control_only") is not control
            or appearance.get("visual_descriptor_values_included") is not (not control)
            or appearance.get("topology_or_original_crop_values_included") is not control
            or not isinstance(contract, Mapping)
            or contract.get("mode")
            != "partial_center_excluded_sfm_maplet_transport_v2"
            or contract.get("support_coordinate_source") != "sfm_observation_xy"
            or contract.get("support_neighbor_scope")
            != "same_real_support_image_only"
            or contract.get("view_aggregation")
            != "none_before_learned_logsumexp_mixture_v1"
            or contract.get("center_descriptor_in_features") is not False
            or contract.get("partial_support_quadrants_retained") is not True
            or contract.get("minimum_total_neighbors")
            != SPARSE_MAPLET_MINIMUM_TOTAL_NEIGHBORS
            or contract.get("maximum_neighbors_per_quadrant")
            != SPARSE_MAPLET_MAX_NEIGHBORS_PER_QUADRANT
            or contract.get("visual_border_padding")
            != "reflection_from_real_feature_map_v1"
            or contract.get("original_crop_topology_control_exported_separately")
            is not True
            or contract.get("availability_is_a_visual_feature") is not False
            or not isinstance(contract.get("neighbor_topology_sha256"), Mapping)
            or len(contract["neighbor_topology_sha256"])
            != len(SPARSE_MAPLET_TRANSPORT_PROFILES)
            or not isinstance(profiles, list)
            or len(profiles) != len(expected_profile_contract)
        ):
            raise ValueError(f"{path}: sparse-maplet metadata contract is invalid")
        for expected, actual in zip(expected_profile_contract, profiles):
            if (
                not isinstance(actual, Mapping)
                or any(actual.get(key) != value for key, value in expected.items())
                or tuple(actual.get("visual_feature_names", ()))
                != SPARSE_MAPLET_TRANSPORT_FEATURE_NAMES_BY_PROFILE[expected["name"]]
                or tuple(actual.get("topology_control_feature_names", ()))
                != SPARSE_MAPLET_TOPOLOGY_CONTROL_FEATURE_NAMES_BY_PROFILE[
                    expected["name"]
                ]
            ):
                raise ValueError(f"{path}: sparse-maplet profile contract is invalid")


def _assert_paired_csr_exact(
    *,
    visual_paths: Sequence[Path],
    control_paths: Sequence[Path],
    visual_headers: Sequence[_StreamingArtifactHeader],
    control_headers: Sequence[_StreamingArtifactHeader],
) -> None:
    """Verify control has no opportunity to exploit a different candidate set."""

    if not (
        len(visual_paths)
        == len(control_paths)
        == len(visual_headers)
        == len(control_headers)
    ):
        raise ValueError("sparse-maplet visual/control shard counts differ")
    common_fields = (
        "verification_query_ids",
        "split_names",
        "verification_source_row_indices",
        "verification_xy",
        "candidate_track_ids",
        "candidate_probabilities",
        "null_probabilities",
        "candidate_support_observation_counts",
        "source_maplet_support_view_counts",
        "edge_candidate_offsets",
        "edge_geometry_rows",
        "edge_sparse_maplet_support_quadrant_counts",
    )
    for visual_path, control_path, visual_header, control_header in zip(
        visual_paths, control_paths, visual_headers, control_headers
    ):
        if (
            visual_header.query_ids.tolist() != control_header.query_ids.tolist()
            or visual_header.split_names.tolist() != control_header.split_names.tolist()
            or not np.array_equal(
                visual_header.source_row_indices, control_header.source_row_indices
            )
            or not np.array_equal(
                visual_header.candidate_track_ids, control_header.candidate_track_ids
            )
            or not np.array_equal(
                visual_header.candidate_probabilities,
                control_header.candidate_probabilities,
            )
            or not np.array_equal(
                visual_header.candidate_support_observation_counts,
                control_header.candidate_support_observation_counts,
            )
            or not np.array_equal(
                visual_header.edge_candidate_offsets, control_header.edge_candidate_offsets
            )
            or int(visual_header.edge_count) != int(control_header.edge_count)
        ):
            raise ValueError("sparse-maplet visual/control CSR headers differ")
        with np.load(visual_path, allow_pickle=False) as visual, np.load(
            control_path, allow_pickle=False
        ) as control:
            if any(
                field not in visual.files
                or field not in control.files
                or not np.array_equal(visual[field], control[field])
                for field in common_fields
            ):
                raise ValueError("sparse-maplet visual/control frozen CSR differs")
        visual_contract = visual_header.metadata["sparse_maplet_transport_contract"]
        control_contract = control_header.metadata["sparse_maplet_transport_contract"]
        if (
            visual_contract.get("neighbor_topology_sha256")
            != control_contract.get("neighbor_topology_sha256")
            or visual_header.metadata.get("source_fulltrack_per_view_artifact_sha256")
            != control_header.metadata.get("source_fulltrack_per_view_artifact_sha256")
        ):
            raise ValueError("sparse-maplet visual/control lineage differs")


def _profile_group_indices(*, family: str, control: bool) -> tuple[int, ...]:
    spec = FULLTRACK_PER_VIEW_FAMILIES[str(family)]
    expected_semantics = (
        FULLTRACK_PER_VIEW_SPARSE_MAPLET_TOPOLOGY_CONTROL_EDGE_FEATURE_SEMANTICS
        if control
        else FULLTRACK_PER_VIEW_SPARSE_MAPLET_TRANSPORT_EDGE_FEATURE_SEMANTICS
    )
    if spec.edge_feature_semantics != expected_semantics:
        raise ValueError("sparse-maplet family edge semantics differ from its role")
    selected = tuple(spec.profile_names)
    names_by_profile = (
        SPARSE_MAPLET_TOPOLOGY_CONTROL_FEATURE_NAMES_BY_PROFILE
        if control
        else SPARSE_MAPLET_TRANSPORT_FEATURE_NAMES_BY_PROFILE
    )
    groups: list[int] = []
    expected_columns: list[str] = []
    for index, profile in enumerate(SPARSE_MAPLET_TRANSPORT_PROFILES):
        names = tuple(names_by_profile[profile.name])
        overlap = set(names).intersection(selected)
        if overlap and overlap != set(names):
            raise ValueError("sparse-maplet family selects a partial profile")
        if overlap:
            groups.append(index)
            expected_columns.extend(names)
    if tuple(expected_columns) != selected or not groups:
        raise ValueError("sparse-maplet family profiles are not a complete ordered group set")
    return tuple(groups)


def _initial_totals() -> dict[str, float | int]:
    return {
        "candidate_count": 0,
        "candidate_prior_mass": 0.0,
        "candidate_with_usable_sparse_maplet_edge_count": 0,
        "candidate_prior_mass_covered": 0.0,
        "real_support_view_sum": 0.0,
        "usable_sparse_maplet_view_sum": 0.0,
        "edge_count": 0,
        "usable_sparse_maplet_edge_count": 0,
        "usable_edge_selected_profile_total_neighbor_sum": 0.0,
        "usable_edge_selected_profile_min_neighbor_sum": 0.0,
        "usable_edge_selected_profile_populated_quadrant_sum": 0.0,
    }


def _stream_coverage(
    *,
    visual_paths: Sequence[Path],
    control_paths: Sequence[Path],
    visual_headers: Sequence[_StreamingArtifactHeader],
    control_headers: Sequence[_StreamingArtifactHeader],
    visual_families: Sequence[str],
) -> list[dict[str, Any]]:
    if not visual_headers:
        raise ValueError("sparse-maplet coverage needs non-empty headers")
    feature_indices = {
        family: profile_indices_for_fulltrack_per_view_family(
            family, profile_names=visual_headers[0].profile_names
        )
        for family in visual_families
    }
    control_indices = {
        family: profile_indices_for_fulltrack_per_view_family(
            FULLTRACK_PER_VIEW_SPARSE_MAPLET_PAIRED_CONTROL_FAMILIES[family],
            profile_names=control_headers[0].profile_names,
        )
        for family in visual_families
    }
    groups = {family: _profile_group_indices(family=family, control=False) for family in visual_families}
    for family in visual_families:
        control_family = FULLTRACK_PER_VIEW_SPARSE_MAPLET_PAIRED_CONTROL_FAMILIES[family]
        if groups[family] != _profile_group_indices(family=control_family, control=True):
            raise ValueError("sparse-maplet visual/control profile groups differ")
    totals = {
        (family, split, bucket): _initial_totals()
        for family in visual_families
        for split in EXPECTED_QUERY_COUNTS
        for bucket, _minimum, _maximum in RANK_BUCKETS
    }
    for visual_path, control_path, header in zip(visual_paths, control_paths, visual_headers):
        ranks = _frozen_candidate_ranks(
            header.candidate_track_ids, header.candidate_probabilities
        )
        candidate_count = int(header.candidate_track_ids.shape[1])
        counts = np.asarray(header.candidate_support_observation_counts, dtype=np.int64)
        edge_candidates = np.repeat(
            np.arange(counts.size, dtype=np.int64), counts.reshape(-1)
        )
        if len(edge_candidates) != int(header.edge_count):
            raise ValueError(f"{visual_path}: sparse-maplet CSR edge count is inconsistent")
        candidate_rows = np.repeat(
            np.arange(len(header.query_ids), dtype=np.int64), candidate_count
        )
        edge_rows = edge_candidates // candidate_count
        candidate_ranks = ranks.reshape(-1)
        edge_ranks = candidate_ranks[edge_candidates]
        candidate_mass = np.asarray(
            header.candidate_probabilities, dtype=np.float32
        ).reshape(-1)
        support_counts = counts.reshape(-1)
        with np.load(visual_path, allow_pickle=False) as visual, np.load(
            control_path, allow_pickle=False
        ) as control:
            visual_valid = np.asarray(visual["edge_profile_valid"], dtype=bool)
            control_valid = np.asarray(control["edge_profile_valid"], dtype=bool)
            neighbor_counts = np.asarray(
                visual["edge_sparse_maplet_support_quadrant_counts"], dtype=np.uint8
            )
        if neighbor_counts.shape != (
            int(header.edge_count),
            len(SPARSE_MAPLET_TRANSPORT_PROFILES),
            4,
        ):
            raise ValueError(f"{visual_path}: sparse-maplet neighbour-count shape is invalid")
        for family in visual_families:
            edge_valid = np.all(visual_valid[:, feature_indices[family]], axis=1)
            paired_valid = np.all(control_valid[:, control_indices[family]], axis=1)
            if not np.array_equal(edge_valid, paired_valid):
                raise ValueError("sparse-maplet visual/control selected-edge coverage differs")
            candidate_usable_views = np.zeros((counts.size,), dtype=np.int64)
            np.add.at(candidate_usable_views, edge_candidates, edge_valid.astype(np.int64))
            selected_counts = neighbor_counts[:, groups[family], :].astype(np.int64)
            selected_total = selected_counts.sum(axis=2)
            selected_min_total = selected_total.min(axis=1)
            selected_mean_total = selected_total.mean(axis=1)
            selected_mean_populated = (selected_counts > 0).sum(axis=2).mean(axis=1)
            for split in EXPECTED_QUERY_COUNTS:
                row_mask = header.split_names == split
                candidate_split = row_mask[candidate_rows]
                edge_split = row_mask[edge_rows]
                for bucket, minimum_rank, maximum_rank in RANK_BUCKETS:
                    total = totals[(family, split, bucket)]
                    candidate_mask = (
                        candidate_split
                        & (candidate_ranks >= int(minimum_rank))
                        & (candidate_ranks <= int(maximum_rank))
                    )
                    flat = np.flatnonzero(candidate_mask)
                    usable_candidate = candidate_usable_views[flat] > 0
                    masses = candidate_mass[flat]
                    total["candidate_count"] = int(total["candidate_count"]) + int(len(flat))
                    total["candidate_prior_mass"] = float(total["candidate_prior_mass"]) + float(
                        np.sum(masses, dtype=np.float64)
                    )
                    total["candidate_with_usable_sparse_maplet_edge_count"] = int(
                        total["candidate_with_usable_sparse_maplet_edge_count"]
                    ) + int(np.sum(usable_candidate))
                    total["candidate_prior_mass_covered"] = float(
                        total["candidate_prior_mass_covered"]
                    ) + float(np.sum(masses[usable_candidate], dtype=np.float64))
                    total["real_support_view_sum"] = float(total["real_support_view_sum"]) + float(
                        np.sum(support_counts[flat], dtype=np.float64)
                    )
                    total["usable_sparse_maplet_view_sum"] = float(
                        total["usable_sparse_maplet_view_sum"]
                    ) + float(np.sum(candidate_usable_views[flat], dtype=np.float64))
                    edge_mask = (
                        edge_split
                        & (edge_ranks >= int(minimum_rank))
                        & (edge_ranks <= int(maximum_rank))
                    )
                    usable_edge_mask = edge_mask & edge_valid
                    total["edge_count"] = int(total["edge_count"]) + int(np.sum(edge_mask))
                    total["usable_sparse_maplet_edge_count"] = int(
                        total["usable_sparse_maplet_edge_count"]
                    ) + int(np.sum(usable_edge_mask))
                    total["usable_edge_selected_profile_total_neighbor_sum"] = float(
                        total["usable_edge_selected_profile_total_neighbor_sum"]
                    ) + float(np.sum(selected_mean_total[usable_edge_mask], dtype=np.float64))
                    total["usable_edge_selected_profile_min_neighbor_sum"] = float(
                        total["usable_edge_selected_profile_min_neighbor_sum"]
                    ) + float(np.sum(selected_min_total[usable_edge_mask], dtype=np.float64))
                    total["usable_edge_selected_profile_populated_quadrant_sum"] = float(
                        total["usable_edge_selected_profile_populated_quadrant_sum"]
                    ) + float(np.sum(selected_mean_populated[usable_edge_mask], dtype=np.float64))
        del visual_valid, control_valid, neighbor_counts
    rows: list[dict[str, Any]] = []
    for family in visual_families:
        control_family = FULLTRACK_PER_VIEW_SPARSE_MAPLET_PAIRED_CONTROL_FAMILIES[family]
        for split in EXPECTED_QUERY_COUNTS:
            for bucket, _minimum, _maximum in RANK_BUCKETS:
                total = totals[(family, split, bucket)]
                candidate_total = int(total["candidate_count"])
                edge_total = int(total["edge_count"])
                usable_edge_total = int(total["usable_sparse_maplet_edge_count"])
                mass_total = float(total["candidate_prior_mass"])
                rows.append(
                    {
                        "visual_family": family,
                        "matched_topology_control_family": control_family,
                        "profile_group_count": int(len(groups[family])),
                        "split": split,
                        "rank_bucket": bucket,
                        "candidate_count": candidate_total,
                        "candidate_prior_mass": mass_total,
                        "candidate_with_usable_sparse_maplet_edge_count": int(
                            total["candidate_with_usable_sparse_maplet_edge_count"]
                        ),
                        "candidate_coverage_rate": (
                            int(total["candidate_with_usable_sparse_maplet_edge_count"])
                            / candidate_total
                            if candidate_total
                            else None
                        ),
                        "candidate_prior_mass_covered": float(
                            total["candidate_prior_mass_covered"]
                        ),
                        "candidate_prior_mass_coverage_rate": (
                            float(total["candidate_prior_mass_covered"]) / mass_total
                            if mass_total > 0.0
                            else None
                        ),
                        "mean_real_support_view_count": (
                            float(total["real_support_view_sum"]) / candidate_total
                            if candidate_total
                            else None
                        ),
                        "mean_usable_sparse_maplet_view_count": (
                            float(total["usable_sparse_maplet_view_sum"]) / candidate_total
                            if candidate_total
                            else None
                        ),
                        "edge_count": edge_total,
                        "usable_sparse_maplet_edge_count": usable_edge_total,
                        "joint_edge_coverage_rate": (
                            usable_edge_total / edge_total if edge_total else None
                        ),
                        "missing_edge_rate": (
                            1.0 - usable_edge_total / edge_total if edge_total else None
                        ),
                        "mean_selected_profile_total_neighbors_on_usable_edge": (
                            float(total["usable_edge_selected_profile_total_neighbor_sum"])
                            / usable_edge_total
                            if usable_edge_total
                            else None
                        ),
                        "mean_selected_profile_min_neighbors_on_usable_edge": (
                            float(total["usable_edge_selected_profile_min_neighbor_sum"])
                            / usable_edge_total
                            if usable_edge_total
                            else None
                        ),
                        "mean_selected_profile_populated_quadrants_on_usable_edge": (
                            float(total["usable_edge_selected_profile_populated_quadrant_sum"])
                            / usable_edge_total
                            if usable_edge_total
                            else None
                        ),
                    }
                )
    return rows


def audit_frozen_fulltrack_per_view_sparse_maplet_transport_coverage(
    *,
    export_manifest: Path,
    projected_landmark_bank: Path,
    output_dir: Path,
    visual_families: Sequence[str],
) -> dict[str, Any]:
    """Run the strict full-export coverage and paired-control audit."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite sparse-maplet coverage audit")
    visual_paths, control_paths = _load_completed_export_paths(Path(export_manifest))
    visual_headers = _load_streaming_headers(visual_paths)
    control_headers = _load_streaming_headers(control_paths)
    _validate_sparse_maplet_metadata(paths=visual_paths, headers=visual_headers, control=False)
    _validate_sparse_maplet_metadata(paths=control_paths, headers=control_headers, control=True)
    _assert_paired_csr_exact(
        visual_paths=visual_paths,
        control_paths=control_paths,
        visual_headers=visual_headers,
        control_headers=control_headers,
    )
    _validate_projected_bank_lineage(
        tuple(header.metadata for header in visual_headers),
        projected_landmark_bank=Path(projected_landmark_bank),
    )
    rows = _stream_coverage(
        visual_paths=visual_paths,
        control_paths=control_paths,
        visual_headers=visual_headers,
        control_headers=control_headers,
        visual_families=tuple(visual_families),
    )
    output.mkdir(parents=True, exist_ok=False)
    with (output / "rank_bucket_coverage.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "stage": "audit_frozen_fulltrack_per_view_sparse_maplet_transport_coverage",
        "format": ARTIFACT_FORMAT,
        "export_manifest": {
            "path": str(Path(export_manifest).resolve()),
            "sha256": file_sha256_short(Path(export_manifest)),
        },
        "projected_landmark_bank": {
            "path": str(Path(projected_landmark_bank).resolve()),
            "sha256": file_sha256_short(Path(projected_landmark_bank)),
        },
        "visual_artifact_count": int(len(visual_paths)),
        "topology_control_artifact_count": int(len(control_paths)),
        "visual_families": list(visual_families),
        "paired_control_families": {
            family: FULLTRACK_PER_VIEW_SPARSE_MAPLET_PAIRED_CONTROL_FAMILIES[family]
            for family in visual_families
        },
        "row_count": int(sum(len(header.query_ids) for header in visual_headers)),
        "edge_count": int(sum(header.edge_count for header in visual_headers)),
        "rank_buckets": [name for name, _minimum, _maximum in RANK_BUCKETS],
        "coverage_rows": rows,
        "protocol": {
            "target_free": True,
            "identity_or_pose_targets_loaded": False,
            "edge_profile_scores_loaded": False,
            "fixed_global_top_l": 20,
            "candidate_reselection": False,
            "support_reselection": False,
            "all_real_sfm_support_observations_retained": True,
            "support_view_features_averaged_before_inference": False,
            "partial_center_excluded_sfm_maplets_retained": True,
            "minimum_total_neighbors": SPARSE_MAPLET_MINIMUM_TOTAL_NEIGHBORS,
            "visual_and_control_csr_exactly_paired": True,
            "visual_and_control_selected_edge_validity_exactly_paired": True,
            "image_retrieval_or_submap_used": False,
            "render": False,
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = audit_frozen_fulltrack_per_view_sparse_maplet_transport_coverage(
        export_manifest=Path(args.export_manifest),
        projected_landmark_bank=Path(args.projected_landmark_bank),
        output_dir=Path(args.output_dir),
        visual_families=_visual_families(str(args.visual_families)),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
