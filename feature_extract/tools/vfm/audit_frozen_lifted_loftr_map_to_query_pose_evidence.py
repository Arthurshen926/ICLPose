"""Target-side paired audit for fixed-maplet lifted LoFTR P1 evidence.

The matching scorer deliberately never opens a target artifact and never fits
or updates PnP.  This companion is the first and only target consumer.  It
checks that visual and XYZ-permutation-control runs have exactly the same
frozen hypothesis layout and query evidence before joining pose errors.

Raw LoFTR scores are diagnostic only.  Passing this audit authorizes at most a
separate train-only likelihood calibration experiment; it never promotes an
uncalibrated score into pose selection or PnP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.tools.vfm.audit_frozen_multiscale_candidate_pose_evidence import (
    TARGET_FORMAT,
    _baseline_rows,
    _paired,
    _per_query_rows,
    _split_summary,
    _validate_alpha_zero,
    _write_csv,
)
from feature_extract.tools.vfm.audit_v5_dynamic_absolute_context_pose_evidence import (
    _load_targets,
    _row_keys,
)
from feature_extract.tools.vfm.score_frozen_lifted_loftr_map_to_query_pose_evidence import (
    EVIDENCE_LAYOUT_CANDIDATE_SUPPORT_VIEW,
    EVIDENCE_LAYOUT_GLOBAL_TRACK_UNION,
    SCORE_FORMAT,
    SCORE_VERSION,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.frozen_loftr_coordinate_contract import (
    validate_frozen_loftr_colmap_coordinate_metadata,
)


_EPSILON = 1e-12
_STATISTIC_FIELDS = {
    "mean": "profile_log_likelihood_means",
    "median": "profile_log_likelihood_medians",
    "worst_quartile_mean": "profile_log_likelihood_worst_quartile_means",
    "spatial_median_of_means_2x2": "profile_spatial_median_of_means_2x2",
}
_ROW_FIELDS = (
    "query_ids",
    "split_names",
    "evaluation_labels",
    "hypothesis_indices",
    "source_chosen_for_optional_pose",
    "baseline_score_top1",
    "baseline_selection_scores",
    "profile_log_likelihood_means",
    "profile_log_likelihood_medians",
    "profile_log_likelihood_worst_quartile_means",
    "profile_spatial_median_of_means_2x2",
    "profile_projection_visible_fractions",
)
_GLOBAL_BANK_FIELDS = (
    "fit_query_xy",
    "candidate_group_track_indices",
    "candidate_group_reference_xy",
    "candidate_group_active_mask",
    "candidate_group_identity_probabilities",
    "candidate_group_null_probabilities",
    "canonical_track_ids",
    "canonical_xyz",
    "mode_offsets",
    "mode_query_xy",
    "mode_weights",
    "mode_support_image_ids",
    "mode_support_view_counts",
    "mode_confidence_sums",
    "track_reliabilities",
    "track_reference_xy",
    "active_xyz",
)
_CANDIDATE_SUPPORT_VIEW_BANK_FIELDS = (
    "fit_query_xy",
    "candidate_group_track_indices",
    "candidate_group_reference_xy",
    "candidate_group_active_mask",
    "candidate_group_identity_probabilities",
    "candidate_group_null_probabilities",
    "candidate_group_support_slot_indices",
    "candidate_group_support_image_ids",
    "candidate_group_support_view_probabilities",
    "canonical_track_ids",
    "canonical_xyz",
    "support_slot_track_indices",
    "support_slot_image_ids",
    "mode_offsets",
    "mode_query_xy",
    "mode_weights",
    "mode_confidence_sums",
    "support_reliabilities",
    "support_reference_xy",
    "active_xyz",
)
_LEGACY_SCORE_VERSIONS = frozenset(
    {"p1_fixed_maplet_union_crossview_lifted_loftr_map_to_query_v5_fixed_identity_prior"}
)
_CANDIDATE_SUPPORT_VIEW_CONTRACT_KEYS = {
    "candidate_specific_support_view_posterior",
    "candidate_support_view_posterior_fixed_before_pose_scoring",
    "candidate_support_view_posterior_target_free",
    "missing_support_view_evidence_is_neutral_ratio",
}
_REQUIRED_STRICT_CONTRACT = {
    "heldout_query_image_content_excludes_pnp_fit_neighborhoods": True,
    "fixed_pnp_fit_topl_maplet_union": True,
    "fixed_global_topl": True,
    "candidate_identity_fixed_across_hypotheses": True,
    "candidate_group_latent_identity_marginalized": True,
    "candidate_group_topl_denominator_fixed": True,
    "candidate_group_explicit_null": True,
    "candidate_group_identity_prior_fixed_before_pose_scoring": True,
    "candidate_group_identity_prior_target_free": True,
    "candidate_groups_without_lifted_evidence_fixed_null_only": True,
    "candidate_group_active_mask_fixed_across_hypotheses": True,
    "pnp_query_center_used_for_group_scoring": False,
    "support_maplet_prefix_fixed": True,
    "support_view_descriptor_averaging": False,
    "support_view_endpoints_averaged_before_likelihood": False,
    "support_view_endpoint_mixture_explicit": True,
    "cross_view_modes_marginalized_not_argmaxed": True,
    "query_center_used_for_loftr_mode_selection": False,
    "pose_dependent_correspondence_selection": False,
    "pose_or_ground_truth_used_for_scoring": False,
    "full_mapping_image_pair_cache": True,
    "image_retrieval_or_submap_used": False,
    "render": False,
    "out_of_image_projection_is_negative_likelihood": True,
    "raw_scores_calibrated_or_promoted": False,
    "raw_scores_must_not_feed_pnp": True,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual_score_artifacts", required=True)
    parser.add_argument("--control_score_artifacts", required=True)
    parser.add_argument("--target_artifact", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--score_statistic",
        choices=tuple(_STATISTIC_FIELDS),
        default="spatial_median_of_means_2x2",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _paths(value: str) -> tuple[Path, ...]:
    paths = tuple(Path(item.strip()) for item in str(value).split(",") if item.strip())
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("lifted LoFTR audit paths must be non-empty and unique")
    return paths


def _canonical_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def _as_metadata(path: Path, payload: Mapping[str, np.ndarray]) -> dict[str, object]:
    if "metadata_json" not in payload:
        raise ValueError(f"{path}: lifted LoFTR score lacks metadata_json")
    try:
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}: lifted LoFTR metadata is malformed") from error
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: lifted LoFTR metadata must be an object")
    return metadata


def _array_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return np.array_equal(np.asarray(left), np.asarray(right), equal_nan=False)


def _evidence_layout(metadata: Mapping[str, object]) -> str:
    layout = str(metadata.get("evidence_layout", EVIDENCE_LAYOUT_GLOBAL_TRACK_UNION))
    if layout not in {
        EVIDENCE_LAYOUT_GLOBAL_TRACK_UNION,
        EVIDENCE_LAYOUT_CANDIDATE_SUPPORT_VIEW,
    }:
        raise ValueError("lifted LoFTR score declares an unknown evidence layout")
    return layout


def _bank_fields_for_layout(layout: str) -> tuple[str, ...]:
    if str(layout) == EVIDENCE_LAYOUT_GLOBAL_TRACK_UNION:
        return _GLOBAL_BANK_FIELDS
    if str(layout) == EVIDENCE_LAYOUT_CANDIDATE_SUPPORT_VIEW:
        return _CANDIDATE_SUPPORT_VIEW_BANK_FIELDS
    raise ValueError("lifted LoFTR score declares an unknown evidence layout")


def _normalised_strict_contract(metadata: Mapping[str, object]) -> dict[str, object]:
    strict = metadata.get("strict_frozen_lifted_map_to_query_contract")
    if not isinstance(strict, Mapping):
        raise ValueError("lifted LoFTR score lacks its strict frozen contract")
    output = {str(key): value for key, value in strict.items()}
    control = output.pop("xyz_permutation_control", None)
    if control not in (True, False):
        raise ValueError("lifted LoFTR score declares an invalid control flag")
    evidence_layout = _evidence_layout(metadata)
    candidate_values = {
        key: output.pop(key, None) for key in _CANDIDATE_SUPPORT_VIEW_CONTRACT_KEYS
    }
    if evidence_layout == EVIDENCE_LAYOUT_CANDIDATE_SUPPORT_VIEW:
        if any(value is not True for value in candidate_values.values()):
            raise ValueError("candidate support-view contract is incomplete")
    elif any(value not in (None, False) for value in candidate_values.values()):
        raise ValueError("global track-union score declares candidate-view semantics")
    if any(output.get(key) is not value for key, value in _REQUIRED_STRICT_CONTRACT.items()):
        raise ValueError("lifted LoFTR score strict contract is incomplete")
    if set(output) != set(_REQUIRED_STRICT_CONTRACT):
        raise ValueError("lifted LoFTR score strict contract contains unknown semantics")
    return output


def _validate_score_metadata(
    *, path: Path, arrays: Mapping[str, np.ndarray], metadata: Mapping[str, object], expected_variant: str
) -> None:
    if (
        metadata.get("format") != SCORE_FORMAT
        or metadata.get("score_version")
        not in ({SCORE_VERSION} | _LEGACY_SCORE_VERSIONS)
        or metadata.get("contains_target_fields") is not False
        or metadata.get("pose_or_ground_truth_used_for_scoring") is not False
        or metadata.get("supervision_arrays_loaded") is not False
        or metadata.get("diagnostic_only") is not True
        or metadata.get("promotion_allowed") is not False
        or metadata.get("evidence_variant") != expected_variant
    ):
        raise ValueError(f"{path}: not a target-free lifted LoFTR P1 score")
    evidence_layout = _evidence_layout(metadata)
    if (
        evidence_layout == EVIDENCE_LAYOUT_CANDIDATE_SUPPORT_VIEW
        and metadata.get("score_version") != SCORE_VERSION
    ):
        raise ValueError(f"{path}: candidate-view scores require the v6 schema")
    _normalised_strict_contract(metadata)
    strict = metadata["strict_frozen_lifted_map_to_query_contract"]
    if bool(strict["xyz_permutation_control"]) != (
        expected_variant == "xyz_permutation_control"
    ):
        raise ValueError(f"{path}: lifted LoFTR control flag and evidence variant disagree")
    runtime = metadata.get("runtime")
    if (
        not isinstance(runtime, Mapping)
        or int(runtime.get("hypothesis_limit", -1)) != 0
        or runtime.get("all_frozen_hypotheses", True) is not True
        or runtime.get("diagnostic_hypothesis_selection", False) is not False
        or runtime.get("diagnostic_group_terms_dumped", False) is not False
    ):
        raise ValueError(f"{path}: lifted LoFTR audit requires every frozen hypothesis")
    if int(metadata.get("row_count", -1)) <= 0:
        raise ValueError(f"{path}: lifted LoFTR score row count is invalid")
    coordinate = metadata.get("coordinate_contract")
    if not isinstance(coordinate, Mapping):
        raise ValueError(f"{path}: lifted LoFTR score has no coordinate contract")
    validate_frozen_loftr_colmap_coordinate_metadata(metadata)
    profiles = metadata.get("score_profiles")
    names = np.asarray(arrays["profile_names"]).astype(str).reshape(-1)
    if (
        not isinstance(profiles, list)
        or len(names) == 0
        or len(set(names.tolist())) != len(names)
        or len(profiles) != len(names)
        or [str(item.get("name", "")) for item in profiles if isinstance(item, Mapping)]
        != names.tolist()
    ):
        raise ValueError(f"{path}: lifted LoFTR score profiles are invalid")
    candidate_union = metadata.get("candidate_maplet_union")
    lifting = metadata.get("lifting_config")
    inputs = metadata.get("inputs")
    if (
        not isinstance(candidate_union, Mapping)
        or int(candidate_union.get("fixed_candidate_top_k", -1)) != 20
        or int(candidate_union.get("fixed_support_view_count", -1)) <= 0
        or not str(candidate_union.get("candidate_union_digest", ""))
        or int(candidate_union.get("candidate_group_count", -1)) != 128
        or int(candidate_union.get("candidate_group_active_count", -1)) <= 0
        or int(candidate_union.get("candidate_group_top_k", -1)) != 20
        or not str(candidate_union.get("candidate_group_digest", ""))
        or not isinstance(lifting, Mapping)
        or lifting.get("support_anchor_mode_selection") != "support_only_no_query_center_v1"
        or not isinstance(inputs, Mapping)
    ):
        raise ValueError(f"{path}: lifted LoFTR provenance is incomplete")


def _load_score(
    path: Path, *, expected_variant: str
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(Path(path), allow_pickle=False) as payload:
        if "metadata_json" not in payload.files:
            raise ValueError(f"{path}: lifted LoFTR score lacks metadata_json")
        metadata_probe = {"metadata_json": np.asarray(payload["metadata_json"]).copy()}
        metadata = _as_metadata(path, metadata_probe)
        bank_fields = _bank_fields_for_layout(_evidence_layout(metadata))
        fields = (*_ROW_FIELDS, "profile_names", *bank_fields, "metadata_json")
        missing = sorted(set(fields).difference(payload.files))
        if missing:
            raise ValueError(f"{path}: lifted LoFTR score lacks {missing}")
        arrays = {field: np.asarray(payload[field]).copy() for field in fields}
    _validate_score_metadata(
        path=Path(path), arrays=arrays, metadata=metadata, expected_variant=expected_variant
    )
    count = int(metadata["row_count"])
    for field in _ROW_FIELDS:
        value = np.asarray(arrays[field])
        if value.shape[0] != count:
            raise ValueError(f"{path}: {field} is not row aligned")
    names = np.asarray(arrays["profile_names"]).astype(str).reshape(-1)
    for field in (
        "profile_log_likelihood_means",
        "profile_log_likelihood_medians",
        "profile_log_likelihood_worst_quartile_means",
        "profile_spatial_median_of_means_2x2",
    ):
        value = np.asarray(arrays[field], dtype=np.float64)
        if value.shape != (count, len(names)) or not np.isfinite(value).all():
            raise ValueError(f"{path}: {field} is malformed")
    visible = np.asarray(arrays["profile_projection_visible_fractions"], dtype=np.float64)
    if (
        visible.shape != (count,)
        or not np.isfinite(visible).all()
        or np.any((visible < 0.0) | (visible > 1.0))
    ):
        raise ValueError(f"{path}: projection visibility is malformed")
    if (
        np.asarray(arrays["fit_query_xy"], dtype=np.float32).shape != (128, 2)
        or np.asarray(arrays["candidate_group_track_indices"], dtype=np.int64).shape
        != (128, 20)
        or np.asarray(arrays["candidate_group_reference_xy"], dtype=np.float32).shape
        != (128, 2)
        or np.asarray(arrays["candidate_group_active_mask"], dtype=bool).shape != (128,)
        or np.asarray(arrays["canonical_track_ids"], dtype=np.int64).ndim != 1
        or len(np.asarray(arrays["canonical_track_ids"])) < 1
    ):
        raise ValueError(f"{path}: lifted LoFTR fixed evidence layout is malformed")
    track_count = len(np.asarray(arrays["canonical_track_ids"], dtype=np.int64))
    canonical_track_ids = np.asarray(arrays["canonical_track_ids"], dtype=np.int64)
    if len(np.unique(canonical_track_ids)) != track_count:
        raise ValueError(f"{path}: lifted LoFTR canonical tracks are not unique")
    group_indices = np.asarray(arrays["candidate_group_track_indices"], dtype=np.int64)
    active_groups = np.asarray(arrays["candidate_group_active_mask"], dtype=bool)
    group_probabilities = np.asarray(
        arrays["candidate_group_identity_probabilities"], dtype=np.float32
    )
    group_null = np.asarray(
        arrays["candidate_group_null_probabilities"], dtype=np.float32
    ).reshape(-1)
    if (
        np.any(group_indices < -1)
        or np.any(group_indices >= track_count)
        or group_probabilities.shape != group_indices.shape
        or group_null.shape != (group_indices.shape[0],)
        or not np.isfinite(group_probabilities).all()
        or not np.isfinite(group_null).all()
        or np.any((group_probabilities < 0.0) | (group_probabilities > 1.0))
        or np.any((group_null < 0.0) | (group_null > 1.0))
        or np.any(
            np.abs(group_probabilities.sum(axis=1, dtype=np.float64) + group_null - 1.0)
            > 1e-4
        )
    ):
        raise ValueError(f"{path}: fixed candidate-group layout is malformed")
    if (
        not np.isfinite(np.asarray(arrays["candidate_group_reference_xy"], dtype=np.float32)).all()
        or not np.isfinite(np.asarray(arrays["canonical_xyz"], dtype=np.float32)).all()
        or not np.isfinite(np.asarray(arrays["active_xyz"], dtype=np.float32)).all()
        or not np.any(active_groups)
    ):
        raise ValueError(f"{path}: lifted LoFTR fixed evidence values are malformed")
    evidence_layout = _evidence_layout(metadata)
    mode_offsets = np.asarray(arrays["mode_offsets"], dtype=np.int64).reshape(-1)
    mode_count = int(mode_offsets[-1]) if len(mode_offsets) else -1
    if evidence_layout == EVIDENCE_LAYOUT_GLOBAL_TRACK_UNION:
        if (
            np.any(active_groups != np.any(group_indices >= 0, axis=1))
            or np.asarray(arrays["canonical_xyz"], dtype=np.float32).shape != (track_count, 3)
            or np.asarray(arrays["active_xyz"], dtype=np.float32).shape != (track_count, 3)
            or mode_offsets.shape != (track_count + 1,)
            or mode_offsets[0] != 0
            or mode_count <= 0
            or np.any(np.diff(mode_offsets) <= 0)
            or np.asarray(arrays["mode_query_xy"], dtype=np.float32).shape != (mode_count, 2)
            or np.asarray(arrays["mode_weights"], dtype=np.float32).shape != (mode_count,)
            or np.asarray(arrays["mode_support_image_ids"]).astype(str).shape != (mode_count,)
            or np.asarray(arrays["mode_support_view_counts"], dtype=np.int64).shape
            != (mode_count,)
            or np.asarray(arrays["mode_confidence_sums"], dtype=np.float32).shape
            != (mode_count,)
            or np.asarray(arrays["track_reliabilities"], dtype=np.float32).shape
            != (track_count,)
            or np.asarray(arrays["track_reference_xy"], dtype=np.float32).shape
            != (track_count, 2)
        ):
            raise ValueError(f"{path}: lifted LoFTR global mode bank shape is malformed")
    else:
        slots = np.asarray(arrays["candidate_group_support_slot_indices"], dtype=np.int64)
        support_ids = np.asarray(arrays["candidate_group_support_image_ids"]).astype(str)
        view_probabilities = np.asarray(
            arrays["candidate_group_support_view_probabilities"], dtype=np.float32
        )
        slot_tracks = np.asarray(arrays["support_slot_track_indices"], dtype=np.int64).reshape(-1)
        slot_image_ids = np.asarray(arrays["support_slot_image_ids"]).astype(str).reshape(-1)
        slot_count = len(slot_tracks)
        if (
            np.asarray(arrays["canonical_xyz"], dtype=np.float32).shape != (track_count, 3)
            or np.asarray(arrays["active_xyz"], dtype=np.float32).shape != (track_count, 3)
            or slots.ndim != 3
            or slots.shape[:2] != group_indices.shape
            or slots.shape[2] <= 0
            or support_ids.shape != slots.shape
            or view_probabilities.shape != slots.shape
            or slot_count <= 0
            or slot_image_ids.shape != (slot_count,)
            or np.any(slot_tracks < 0)
            or np.any(slot_tracks >= track_count)
            or np.any(slots < -1)
            or np.any(slots >= slot_count)
            or np.any(support_ids == "")
            or not np.isfinite(view_probabilities).all()
            or np.any((view_probabilities < 0.0) | (view_probabilities > 1.0))
            or np.any(
                np.abs(view_probabilities.sum(axis=2, dtype=np.float64) - 1.0) > 1e-4
            )
            or mode_offsets.shape != (slot_count + 1,)
            or mode_offsets[0] != 0
            or mode_count <= 0
            or np.any(np.diff(mode_offsets) <= 0)
            or np.asarray(arrays["mode_query_xy"], dtype=np.float32).shape != (mode_count, 2)
            or np.asarray(arrays["mode_weights"], dtype=np.float32).shape != (mode_count,)
            or np.asarray(arrays["mode_confidence_sums"], dtype=np.float32).shape
            != (slot_count,)
            or np.asarray(arrays["support_reliabilities"], dtype=np.float32).shape
            != (slot_count,)
            or np.asarray(arrays["support_reference_xy"], dtype=np.float32).shape
            != (slot_count, 2)
        ):
            raise ValueError(
                f"{path}: lifted LoFTR candidate support-view bank shape is malformed"
            )
        valid_slots = slots >= 0
        expected_tracks = np.broadcast_to(group_indices[:, :, None], slots.shape)
        expected_image_ids = np.empty(slots.shape, dtype=slot_image_ids.dtype)
        expected_image_ids[valid_slots] = slot_image_ids[slots[valid_slots]]
        expected_active = np.any(
            valid_slots
            & (group_probabilities[:, :, None] > 0.0)
            & (view_probabilities > 0.0),
            axis=(1, 2),
        )
        weights = np.asarray(arrays["mode_weights"], dtype=np.float32)
        per_slot_mass = np.add.reduceat(weights, mode_offsets[:-1])
        if (
            np.any(valid_slots & (expected_tracks < 0))
            or np.any(slot_tracks[slots[valid_slots]] != expected_tracks[valid_slots])
            or np.any(support_ids[valid_slots] != expected_image_ids[valid_slots])
            or np.any(active_groups != expected_active)
            or not np.isfinite(weights).all()
            or np.any(weights <= 0.0)
            or np.any(np.abs(per_slot_mass - 1.0) > 1e-4)
            or not np.isfinite(
                np.asarray(arrays["support_reliabilities"], dtype=np.float32)
            ).all()
            or np.any(
                (np.asarray(arrays["support_reliabilities"], dtype=np.float32) < 0.0)
                | (np.asarray(arrays["support_reliabilities"], dtype=np.float32) > 1.0)
            )
            or not np.isfinite(
                np.asarray(arrays["support_reference_xy"], dtype=np.float32)
            ).all()
        ):
            raise ValueError(
                f"{path}: lifted LoFTR candidate support-view bank values are malformed"
            )
    keys = _row_keys(arrays)
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path}: lifted LoFTR score repeats a hypothesis row")
    return arrays, metadata


def _static_input_hashes(metadata: Mapping[str, object]) -> dict[str, str]:
    inputs = metadata.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("lifted LoFTR score has no input manifest")
    query_specific = {"hypothesis_artifact", "baseline_score_artifact", "loftr_pair_cache"}
    output: dict[str, str] = {}
    for name, value in inputs.items():
        if str(name) in query_specific:
            continue
        if not isinstance(value, Mapping) or not str(value.get("sha256", "")):
            raise ValueError("lifted LoFTR input manifest is incomplete")
        output[str(name)] = str(value["sha256"])
    return output


def _implementation_hashes(metadata: Mapping[str, object]) -> dict[str, str]:
    """Compare implementation identity without treating invocation paths as config."""

    implementation = metadata.get("implementation")
    if not isinstance(implementation, Mapping):
        raise ValueError("lifted LoFTR score has no implementation manifest")
    hashes = {
        str(name): str(value)
        for name, value in implementation.items()
        if str(name).endswith("_sha256")
    }
    if set(hashes) != {"script_sha256", "lifted_mode_module_sha256"} or any(
        not value for value in hashes.values()
    ):
        raise ValueError("lifted LoFTR implementation manifest is incomplete")
    return hashes


def _merge_scores(
    paths: Sequence[Path], *, expected_variant: str
) -> tuple[dict[str, np.ndarray], list[dict[str, object]], tuple[tuple[str, str, str, int], ...]]:
    loaded = [_load_score(path, expected_variant=expected_variant) for path in paths]
    if not loaded:
        raise ValueError("lifted LoFTR audit needs at least one score artifact")
    first_arrays, first_metadata = loaded[0]
    expected_names = np.asarray(first_arrays["profile_names"]).astype(str)
    compatibility = {
        "score_version": first_metadata.get("score_version"),
        "evidence_layout": _evidence_layout(first_metadata),
        "strict": _normalised_strict_contract(first_metadata),
        "lifting_config": first_metadata.get("lifting_config"),
        "score_profiles": first_metadata.get("score_profiles"),
        "static_input_hashes": _static_input_hashes(first_metadata),
        "implementation": _implementation_hashes(first_metadata),
    }
    fingerprint = _canonical_hash(compatibility)
    parts: dict[str, list[np.ndarray]] = {field: [] for field in _ROW_FIELDS}
    metadata_rows: list[dict[str, object]] = []
    for arrays, metadata in loaded:
        item_compatibility = {
            "score_version": metadata.get("score_version"),
            "evidence_layout": _evidence_layout(metadata),
            "strict": _normalised_strict_contract(metadata),
            "lifting_config": metadata.get("lifting_config"),
            "score_profiles": metadata.get("score_profiles"),
            "static_input_hashes": _static_input_hashes(metadata),
            "implementation": _implementation_hashes(metadata),
        }
        if _canonical_hash(item_compatibility) != fingerprint:
            raise ValueError("lifted LoFTR score shards use incompatible frozen configurations")
        if not _array_equal(arrays["profile_names"], expected_names):
            raise ValueError("lifted LoFTR score shards use different profile orders")
        for field in _ROW_FIELDS:
            parts[field].append(np.asarray(arrays[field]))
        metadata_rows.append(metadata)
    merged = {field: np.concatenate(values, axis=0) for field, values in parts.items()}
    merged["profile_names"] = expected_names.copy()
    keys = _row_keys(merged)
    if len(keys) != len(set(keys)):
        raise ValueError("merged lifted LoFTR score shards repeat hypothesis rows")
    return merged, metadata_rows, keys


def _validate_visual_control_pairing(
    *,
    visual_artifacts: Sequence[tuple[dict[str, np.ndarray], Mapping[str, object]]],
    control_artifacts: Sequence[tuple[dict[str, np.ndarray], Mapping[str, object]]],
) -> None:
    visual = {
        str(metadata.get("query_id")): (arrays, metadata)
        for arrays, metadata in visual_artifacts
    }
    control = {
        str(metadata.get("query_id")): (arrays, metadata)
        for arrays, metadata in control_artifacts
    }
    if not visual or set(visual) != set(control) or len(visual) != len(visual_artifacts):
        raise ValueError("lifted LoFTR visual/control artifacts do not cover identical unique queries")
    static_metadata_keys = (
        "evidence_layout",
        "candidate_maplet_union",
        "lifting_config",
        "coordinate_contract",
        "pair_cache_contract",
        "input_metadata_hashes",
        "canonical_mode_layout_sha256",
        "canonical_xyz_sha256",
        "score_profiles",
    )
    row_fields = (
        "query_ids",
        "split_names",
        "evaluation_labels",
        "hypothesis_indices",
        "source_chosen_for_optional_pose",
        "baseline_score_top1",
        "baseline_selection_scores",
        "profile_names",
    )
    for query_id in sorted(visual):
        visual_arrays, visual_metadata = visual[query_id]
        control_arrays, control_metadata = control[query_id]
        visual_layout = _evidence_layout(visual_metadata)
        control_layout = _evidence_layout(control_metadata)
        canonical_fields = tuple(
            field for field in _bank_fields_for_layout(visual_layout) if field != "active_xyz"
        )
        if (
            visual_metadata.get("evidence_variant") != "visual"
            or control_metadata.get("evidence_variant") != "xyz_permutation_control"
            or visual_layout != control_layout
            or _normalised_strict_contract(visual_metadata)
            != _normalised_strict_contract(control_metadata)
            or any(
                visual_metadata.get(key) != control_metadata.get(key) for key in static_metadata_keys
            )
        ):
            raise ValueError("lifted LoFTR visual/control provenance differs beyond XYZ control")
        if any(
            not _array_equal(visual_arrays[field], control_arrays[field]) for field in row_fields
        ) or any(
            not _array_equal(visual_arrays[field], control_arrays[field]) for field in canonical_fields
        ):
            raise ValueError("lifted LoFTR visual/control layouts differ beyond active XYZ")
        canonical_xyz = np.asarray(visual_arrays["canonical_xyz"], dtype=np.float32)
        if not _array_equal(visual_arrays["active_xyz"], canonical_xyz):
            raise ValueError("visual lifted LoFTR artifact does not use its canonical XYZ")
        permutation = control_metadata.get("xyz_permutation")
        if not isinstance(permutation, list):
            raise ValueError("lifted LoFTR control lacks an XYZ permutation")
        permutation_array = np.asarray(permutation, dtype=np.int64).reshape(-1)
        if (
            permutation_array.shape != (len(canonical_xyz),)
            or sorted(permutation_array.tolist()) != list(range(len(canonical_xyz)))
            or np.any(permutation_array == np.arange(len(canonical_xyz), dtype=np.int64))
            or not _array_equal(control_arrays["active_xyz"], canonical_xyz[permutation_array])
        ):
            raise ValueError("lifted LoFTR XYZ control is not a deterministic derangement")


def _paired_rank(
    visual_rows: Sequence[Mapping[str, object]], control_rows: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    def key(row: Mapping[str, object]) -> tuple[str, str, str]:
        return (str(row["split_name"]), str(row["evaluation_label"]), str(row["query_id"]))

    visual = {key(row): row for row in visual_rows}
    control = {key(row): row for row in control_rows}
    if not visual or set(visual) != set(control):
        raise ValueError("lifted LoFTR visual/control per-query rows are unpaired")
    rank_delta = np.asarray(
        [
            float(visual[item]["oracle_score_rank"]) - float(control[item]["oracle_score_rank"])
            for item in sorted(visual)
        ],
        dtype=np.float64,
    )
    return {
        "oracle_rank_wins": int(np.count_nonzero(rank_delta < -_EPSILON)),
        "oracle_rank_losses": int(np.count_nonzero(rank_delta > _EPSILON)),
        "oracle_rank_ties": int(np.count_nonzero(np.abs(rank_delta) <= _EPSILON)),
        "median_oracle_rank_delta": float(np.median(rank_delta)),
        "mean_oracle_rank_delta": float(np.mean(rank_delta)),
    }


def _primary_gate(
    *, visual: Mapping[str, object], control: Mapping[str, object], baseline: Mapping[str, object], paired_rank: Mapping[str, object]
) -> dict[str, object]:
    visual_rank = float(visual["median_oracle_score_rank"])
    visual_p90_rank = float(visual["p90_oracle_score_rank"])
    visual_p90_translation = float(visual["p90_selected_translation_cm"])
    visual_tail = int(visual["catastrophic_1m_count"])
    signal_checks = {
        "visual_median_oracle_rank_below_control": visual_rank
        < float(control["median_oracle_score_rank"]),
        "visual_p90_oracle_rank_not_worse_than_control": visual_p90_rank
        <= float(control["p90_oracle_score_rank"]),
        "visual_oracle_rank_wins_exceed_losses": int(paired_rank["oracle_rank_wins"])
        > int(paired_rank["oracle_rank_losses"]),
    }
    tail_checks = {
        "visual_p90_oracle_rank_not_worse_than_s0": visual_p90_rank
        <= float(baseline["p90_oracle_score_rank"]),
        "visual_p90_selected_translation_not_worse_than_s0": visual_p90_translation
        <= float(baseline["p90_selected_translation_cm"]),
        "visual_catastrophic_tail_not_worse_than_s0": visual_tail
        <= int(baseline["catastrophic_1m_count"]),
    }
    return {
        **signal_checks,
        **tail_checks,
        "validation_rank_median_under_35": visual_rank < 35.0,
        "raw_p1_gate_passed": bool(
            visual_rank < 35.0 and all(signal_checks.values()) and all(tail_checks.values())
        ),
        "eligible_for_train_only_likelihood_calibration": bool(
            visual_rank < 35.0 and all(signal_checks.values()) and all(tail_checks.values())
        ),
        "eligible_for_pose_or_pnp_promotion": False,
        "reason": (
            "raw lifted LoFTR P1 scores are uncalibrated diagnostic likelihood ratios; "
            "a pass authorizes only train-only calibration and must not alter PnP or selection"
        ),
    }


def _validate_target_lineage(
    *, score_metadata: Sequence[Mapping[str, object]], target_metadata: Mapping[str, object]
) -> None:
    hypotheses = set(str(value) for value in target_metadata.get("hypothesis_artifact_sha256", []))
    baselines = set(str(value) for value in target_metadata.get("score_artifact_sha256", []))
    if not hypotheses or not baselines:
        raise ValueError("target artifact has no frozen S0 lineage")
    for metadata in score_metadata:
        inputs = metadata.get("inputs")
        if not isinstance(inputs, Mapping):
            raise ValueError("lifted LoFTR score lacks target-lineage inputs")
        hypothesis = str((inputs.get("hypothesis_artifact") or {}).get("sha256", ""))
        baseline = str((inputs.get("baseline_score_artifact") or {}).get("sha256", ""))
        if hypothesis not in hypotheses or baseline not in baselines:
            raise ValueError("lifted LoFTR score/target lineage differs from frozen S0")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    summary_path = output_dir / "summary.json"
    if summary_path.exists() and not bool(args.force):
        raise FileExistsError(f"refusing to overwrite existing lifted LoFTR audit: {summary_path}")
    visual_paths = _paths(args.visual_score_artifacts)
    control_paths = _paths(args.control_score_artifacts)
    visual_loaded = [_load_score(path, expected_variant="visual") for path in visual_paths]
    control_loaded = [
        _load_score(path, expected_variant="xyz_permutation_control") for path in control_paths
    ]
    _validate_visual_control_pairing(
        visual_artifacts=visual_loaded, control_artifacts=control_loaded
    )
    visual, visual_metadata, visual_keys = _merge_scores(
        visual_paths, expected_variant="visual"
    )
    control, control_metadata, control_keys = _merge_scores(
        control_paths, expected_variant="xyz_permutation_control"
    )
    if visual_keys != control_keys:
        raise ValueError("lifted LoFTR visual/control score rows have different frozen hypotheses")
    # Target data is intentionally opened only after all target-free layout,
    # provenance, and visual/control tests above have succeeded.
    targets, target_metadata = _load_targets(Path(args.target_artifact))
    if target_metadata.get("format") != TARGET_FORMAT:
        raise ValueError("lifted LoFTR target artifact has an unexpected format")
    _validate_target_lineage(score_metadata=visual_metadata, target_metadata=target_metadata)
    _validate_target_lineage(score_metadata=control_metadata, target_metadata=target_metadata)
    target_positions = {key: row for row, key in enumerate(_row_keys(targets))}
    try:
        target_rows = np.asarray([target_positions[key] for key in visual_keys], dtype=np.int64)
    except KeyError as error:
        raise ValueError("a lifted LoFTR score row is absent from the target artifact") from error
    translation = np.asarray(targets["translation_errors_m"], dtype=np.float64)[target_rows]
    rotation = np.asarray(targets["rotation_errors_deg"], dtype=np.float64)[target_rows]
    baseline = np.asarray(visual["baseline_selection_scores"], dtype=np.float64)
    if (
        not _array_equal(baseline, control["baseline_selection_scores"])
        or not _array_equal(visual["baseline_score_top1"], control["baseline_score_top1"])
        or translation.shape != baseline.shape
        or rotation.shape != baseline.shape
        or not np.isfinite(baseline).all()
    ):
        raise ValueError("lifted LoFTR visual/control baselines or target rows are misaligned")
    tie_orders = np.arange(len(visual_keys), dtype=np.int64)
    baseline_rows = _baseline_rows(
        keys=visual_keys,
        baseline_scores=baseline,
        translation_m=translation,
        rotation_deg=rotation,
        tie_break_orders=tie_orders,
    )
    _validate_alpha_zero(
        source_top1=np.asarray(visual["baseline_score_top1"], dtype=bool),
        keys=visual_keys,
        baseline_rows=baseline_rows,
    )
    statistic_field = _STATISTIC_FIELDS[str(args.score_statistic)]
    visual_values = np.asarray(visual[statistic_field], dtype=np.float64)
    control_values = np.asarray(control[statistic_field], dtype=np.float64)
    profile_names = np.asarray(visual["profile_names"]).astype(str)
    all_rows: list[dict[str, object]] = list(baseline_rows)
    families: dict[str, object] = {}
    for index, name in enumerate(profile_names.tolist()):
        visual_rows = _per_query_rows(
            keys=visual_keys,
            scores=visual_values[:, index],
            baseline_scores=baseline,
            translation_m=translation,
            rotation_deg=rotation,
            tie_break_orders=tie_orders,
            family=f"visual:{name}",
            score_mode=f"uncalibrated_{args.score_statistic}",
            alpha=None,
        )
        control_rows = _per_query_rows(
            keys=visual_keys,
            scores=control_values[:, index],
            baseline_scores=baseline,
            translation_m=translation,
            rotation_deg=rotation,
            tie_break_orders=tie_orders,
            family=f"xyz_permutation_control:{name}",
            score_mode=f"uncalibrated_{args.score_statistic}",
            alpha=None,
        )
        all_rows.extend(visual_rows)
        all_rows.extend(control_rows)
        visual_summary = _split_summary(visual_rows)
        control_summary = _split_summary(control_rows)
        paired_rank = _paired_rank(visual_rows, control_rows)
        family: dict[str, object] = {
            "visual": {
                "splits": visual_summary,
                "paired_vs_s0": _paired(baseline_rows, visual_rows),
            },
            "xyz_permutation_control": {
                "splits": control_summary,
                "paired_vs_s0": _paired(baseline_rows, control_rows),
            },
            "visual_vs_xyz_permutation_control": {
                "paired_rank": paired_rank,
                "paired_selected_translation": _paired(control_rows, visual_rows),
            },
            "effective_evidence": {
                "visual_mean_projection_visible_fraction": float(
                    np.mean(np.asarray(visual["profile_projection_visible_fractions"], dtype=np.float64))
                ),
                "control_mean_projection_visible_fraction": float(
                    np.mean(np.asarray(control["profile_projection_visible_fractions"], dtype=np.float64))
                ),
            },
        }
        if "validation" in visual_summary:
            family["predeclared_validation_gate"] = _primary_gate(
                visual=visual_summary["validation"],
                control=control_summary["validation"],
                baseline=_split_summary(baseline_rows)["validation"],
                paired_rank=paired_rank,
            )
        families[str(name)] = family
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "per_query.csv", all_rows)
    summary: dict[str, Any] = {
        "stage": "frozen_lifted_loftr_map_to_query_pose_target_audit",
        "format": "frozen_lifted_loftr_map_to_query_pose_target_audit_v1",
        "protocol": {
            "target_join_isolated_from_scoring": True,
            "visual_xyz_permutation_control_paired": True,
            "fixed_global_topl": True,
            "fixed_maplet_union": True,
            "heldout_pnp_fit_neighborhoods_excluded": True,
            "cross_view_modes_preserved_and_marginalized": True,
            "candidate_or_support_reselection_per_pose": False,
            "query_center_shortcut": False,
            "no_image_retrieval_or_submap": True,
            "no_render": True,
            "raw_scores_are_uncalibrated": True,
            "promotion_allowed": False,
        },
        "primary_score_statistic": str(args.score_statistic),
        "baseline": {"splits": _split_summary(baseline_rows)},
        "families": families,
        "inputs": {
            "visual_score_artifacts": [str(path) for path in visual_paths],
            "visual_score_artifact_sha256": [file_sha256_short(path) for path in visual_paths],
            "control_score_artifacts": [str(path) for path in control_paths],
            "control_score_artifact_sha256": [file_sha256_short(path) for path in control_paths],
            "target_artifact": str(args.target_artifact),
            "target_artifact_sha256": file_sha256_short(Path(args.target_artifact)),
            "score_contract_hash": _canonical_hash(
                {
                    "strict": _normalised_strict_contract(visual_metadata[0]),
                    "evidence_layout": _evidence_layout(visual_metadata[0]),
                    "lifting_config": visual_metadata[0].get("lifting_config"),
                    "profiles": visual_metadata[0].get("score_profiles"),
                    "static_inputs": _static_input_hashes(visual_metadata[0]),
                    "implementation": visual_metadata[0].get("implementation"),
                }
            ),
        },
        "outputs": {"per_query": str(output_dir / "per_query.csv")},
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
