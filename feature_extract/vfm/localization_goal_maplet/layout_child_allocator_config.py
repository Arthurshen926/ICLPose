"""Signed contract for a route-frozen scene-child layout allocator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from feature_extract.vfm.tokens import compute_file_sha256

from .hierarchical_child_allocator import SEMANTICS as ALLOCATOR_SEMANTICS
from .physical_map import GoalMapletPhysicalMap
from .pure_retrieval import PureRadioPhysicalRetrieval
from .scene_child_evidence import (
    SCENE_CHILD_EVIDENCE_SEMANTICS,
    SCENE_PARENT_MASK_SEMANTICS,
)


CONFIG_SCHEMA = "goal_maplet_layout_child_allocator_config_v1"
SCORE_RAW_EVIDENCE = "raw_child_evidence_v1"
SCORE_EVIDENCE_PER_AREA = "child_evidence_per_surface_area_v1"
SCORE_POLICIES = (SCORE_RAW_EVIDENCE, SCORE_EVIDENCE_PER_AREA)

_SOURCE_SIGNATURE_KEYS = (
    "physical_map_file_sha256",
    "canonical_field_file_sha256",
    "surface_mapper_file_sha256",
    "field_feature_contract_file_sha256",
    "validity_calibration_file_sha256",
    "parent_score_semantics",
    "parent_scene_ranking_semantics",
    "parent_mode_temperature",
    "anonymous_parent_mode_readout_sha256",
    "maximum_parent_candidates",
    "maximum_child_candidates",
    "maximum_scene_parents",
    "child_probability_semantics",
    "token_height",
    "token_width",
    "pool_sizes",
    "pool_weights",
)


def config_content_sha256(payload: Mapping[str, object]) -> str:
    clean = {key: value for key, value in payload.items() if key != "content_sha256"}
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _json_without_duplicates(path: Path) -> dict[str, object]:
    def hook(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    payload = json.loads(path.read_text(), object_pairs_hook=hook)
    if not isinstance(payload, dict):
        raise ValueError("layout allocator JSON root must be an object")
    return payload


def retrieval_layout_source_signature(
    retrieval: PureRadioPhysicalRetrieval,
) -> dict[str, object]:
    metadata = retrieval.metadata
    signature: dict[str, object] = {}
    for key in _SOURCE_SIGNATURE_KEYS:
        if key not in metadata:
            raise ValueError(f"retrieval lacks layout source signature field {key}")
        value = metadata[key]
        if isinstance(value, tuple):
            value = list(value)
        signature[key] = value
    return signature


def _equal_signature_value(left: object, right: object) -> bool:
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, (list, tuple)) or not isinstance(
            right, (list, tuple)
        ):
            return False
        return list(left) == list(right)
    if isinstance(left, (float, np.floating)) or isinstance(right, (float, np.floating)):
        try:
            return bool(np.isclose(float(left), float(right), rtol=0.0, atol=1e-12))
        except (TypeError, ValueError):
            return False
    return left == right


def validate_layout_source_signature(
    tuning_signature: Mapping[str, object],
    deployment_signature: Mapping[str, object],
) -> None:
    for key in _SOURCE_SIGNATURE_KEYS:
        if key not in tuning_signature or key not in deployment_signature:
            raise ValueError(f"layout source signature lacks {key}")
        if not _equal_signature_value(
            tuning_signature[key], deployment_signature[key]
        ):
            raise ValueError(f"layout tuning/deployment signature differs at {key}")


def load_validate_layout_child_allocator_config(
    config_path: Path,
    physical: GoalMapletPhysicalMap,
    *,
    physical_path: Path | None = None,
    query_routes: set[str] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Validate config, its tuning source bytes, and route disjointness.

    Returns the config and the exact retrieval representation signature on
    which it was tuned.  Callers must compare that signature with deployment.
    """

    path = Path(config_path).resolve()
    config = _json_without_duplicates(path)
    if (
        config.get("artifact_type") != CONFIG_SCHEMA
        or config.get("allocator_semantics") != ALLOCATOR_SEMANTICS
        or config.get("candidate_parent_mask_semantics")
        != SCENE_PARENT_MASK_SEMANTICS
        or str(config.get("child_evidence_semantics", ""))
        not in SCENE_CHILD_EVIDENCE_SEMANTICS
        or str(config.get("child_score_policy", "")) not in SCORE_POLICIES
        or str(config.get("physical_map_sha256", "")) != physical.content_sha256
        or str(config.get("content_sha256", ""))
        != config_content_sha256(config)
        or config.get("deployment_uses_gt") is not False
        or config.get("uses_query_pose") is not False
        or config.get("uses_gt_only_for_offline_route_disjoint_config_selection")
        is not True
    ):
        raise ValueError("layout allocator config lineage/claims differ")
    if physical_path is not None and str(
        config.get("physical_map_file_sha256", "")
    ) != compute_file_sha256(Path(physical_path)):
        raise ValueError("layout allocator physical map bytes differ")
    fraction = float(config.get("parent_mass_fraction", -1.0))
    local_block = int(config.get("child_evidence_local_block_size", 0))
    maximum_children = int(config.get("maximum_children", 0))
    maximum_iou = float(config.get("maximum_primitive_iou", -1.0))
    normal_angle = float(config.get("maximum_normal_angle_degrees", -1.0))
    tuning_route = str(config.get("tuning_route", ""))
    if (
        not 0.0 <= fraction <= 1.0
        or local_block <= 0
        or maximum_children <= 0
        or not 0.0 <= maximum_iou <= 1.0
        or not 0.0 <= normal_angle <= 90.0
        or not tuning_route
    ):
        raise ValueError("layout allocator numeric/route contract differs")
    routes = set() if query_routes is None else {str(value) for value in query_routes}
    if tuning_route in routes:
        raise ValueError("layout allocator tuning/query routes overlap")

    binding = config.get("tuning_retrieval")
    if not isinstance(binding, dict):
        raise ValueError("layout allocator lacks tuning retrieval binding")
    paths = [Path(value).resolve() for value in binding.get("retrieval_summary_paths", [])]
    hashes = [str(value) for value in binding.get("retrieval_summary_file_sha256", [])]
    if not paths or len(paths) != len(hashes):
        raise ValueError("layout tuning retrieval inventory differs")
    records: list[dict[str, object]] = []
    for summary_path, expected_hash in zip(paths, hashes):
        if compute_file_sha256(summary_path) != expected_hash:
            raise ValueError("layout tuning retrieval summary bytes differ")
        summary = _json_without_duplicates(summary_path)
        if (
            summary.get("artifact_type")
            != "goal_maplet_pure_radio_retrieval_run_v1"
            or summary.get("control_only") is not True
            or list(
                summary.get("query_split_audit", {}).get(
                    "query_trajectory_ids", []
                )
            ) != [tuning_route]
        ):
            raise ValueError("layout tuning retrieval route/control contract differs")
        records.extend(list(summary.get("rows", [])))
    if not records:
        raise ValueError("layout tuning retrieval is empty")
    first = records[0]
    retrieval_path = Path(str(first.get("artifact", ""))).resolve()
    if compute_file_sha256(retrieval_path) != str(first.get("artifact_sha256", "")):
        raise ValueError("layout tuning retrieval artifact bytes differ")
    retrieval = PureRadioPhysicalRetrieval.load_npz(retrieval_path)
    if retrieval.content_sha256 != str(first.get("content_sha256", "")):
        raise ValueError("layout tuning retrieval artifact content differs")
    signature = retrieval_layout_source_signature(retrieval)
    return config, signature
