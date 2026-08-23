"""Candidate-conditioned soft pose attribution without point correspondences.

Retrieval child probabilities are a pose-free prior.  A basin-specific render
supplies a separate low-dimensional pose field plus normal, relative-depth and
boundary observations.  The kernel conservatively transfers prior mass to a
matched child; every unsupported or incompatible unit becomes explicit
unmatched mass.  There is no per-candidate renormalization that can reward
disappearing evidence, no hard correspondence, and no PnP interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from .pure_retrieval import PureRadioPhysicalRetrieval
from .soft_surface_pose_energy import query_only_pose_reliability_weights


POSE_FIELD_SEMANTICS = "candidate_conditioned_lowd_pose_equivariant_surface_field_v1"
SPARSE_TRANSPORT_SEMANTICS = "candidate_conditioned_sparse_substochastic_pose_transport_v2"


def _sha256(value: str, *, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return text


@dataclass(frozen=True)
class PoseTransportHierarchy:
    child_parent_ids: np.ndarray
    child_support_ids: np.ndarray
    adjacency_offsets: np.ndarray
    adjacency_child_rows: np.ndarray
    content_sha256: str


def pose_transport_hierarchy_content_sha256(
    child_parent_ids: np.ndarray,
    child_support_ids: np.ndarray,
    adjacency_offsets: np.ndarray,
    adjacency_child_rows: np.ndarray,
) -> str:
    """Hash exact hierarchy arrays under the sparse-transport semantics."""

    digest = hashlib.sha256()
    digest.update(SPARSE_TRANSPORT_SEMANTICS.encode("utf-8") + b"\0hierarchy\0")
    for name, source in (
        ("child_parent_ids", child_parent_ids),
        ("child_support_ids", child_support_ids),
        ("adjacency_offsets", adjacency_offsets),
        ("adjacency_child_rows", adjacency_child_rows),
    ):
        value = np.asarray(source, dtype=np.int64).reshape(-1)
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class QueryPoseTransportObservation:
    query_id: str
    radio_content_sha256: str
    readout_content_sha256: str
    pose_codes: np.ndarray
    normals_camera: np.ndarray
    relative_depth: np.ndarray
    boundary: np.ndarray
    pose_code_valid: np.ndarray
    normal_valid: np.ndarray
    depth_valid: np.ndarray
    boundary_valid: np.ndarray
    pose_code_confidence: np.ndarray
    normal_confidence: np.ndarray
    depth_confidence: np.ndarray
    boundary_confidence: np.ndarray
    normal_frame: str = "camera"
    depth_semantics: str = "centered_log_depth_v1"


@dataclass(frozen=True)
class CandidatePoseTransportObservation:
    basin_id: str
    pose_field_content_sha256: str
    child_rows: np.ndarray
    child_weights: np.ndarray
    pose_codes: np.ndarray
    normals_camera: np.ndarray
    double_sided: np.ndarray
    relative_depth: np.ndarray
    boundary: np.ndarray
    pose_code_valid: np.ndarray
    normal_valid: np.ndarray
    depth_valid: np.ndarray
    boundary_valid: np.ndarray
    pose_code_confidence: np.ndarray
    normal_confidence: np.ndarray
    depth_confidence: np.ndarray
    boundary_confidence: np.ndarray
    normal_frame: str = "camera"
    depth_semantics: str = "centered_log_depth_v1"


@dataclass(frozen=True)
class SparsePoseTransportResult:
    source_child_rows: np.ndarray
    source_child_probabilities: np.ndarray
    matched_source_probability: np.ndarray
    unmatched_source_probability: np.ndarray
    token_matched_probability: np.ndarray
    token_score: np.ndarray
    combined_score: float
    edge_count: int
    stage: str
    transport_semantics: str
    query_id: str
    basin_id: str
    query_radio_content_sha256: str
    query_readout_content_sha256: str
    map_pose_field_content_sha256: str
    hierarchy_content_sha256: str
    transport_model_content_sha256: str
    production_eligible: bool


_STAGE = {
    "coarse": {
        "radius": 3, "relations": {"exact", "support", "adjacent", "parent"},
        "depth": "ordinal_depth_v1", "weights": (0.0, 0.7, 0.0, 0.0, 1.2, 0.8),
        "sink": 0.0,
    },
    "medium": {
        "radius": 2, "relations": {"exact", "support", "adjacent"},
        "depth": "centered_log_depth_v1", "weights": (0.5, 0.7, 0.7, 0.3, 1.0, 0.8),
        "sink": 0.0,
    },
    "fine": {
        "radius": 0, "relations": {"exact"},
        "depth": "metric_log_depth_with_uncertainty_v1", "weights": (1.2, 0.5, 0.8, 0.6, 1.0, 0.7),
        "sink": 0.0,
    },
}


def reference_pose_transport_model_content_sha256(
    stage: str, *, depth_scale: float = 0.25, zero_norm_threshold: float = 1.0e-8
) -> str:
    """Hash the complete frozen reference energy configuration.

    A learned transport implementation must replace this reference hash with
    the hash of its serialized parameters and architecture contract.  The
    reference hash prevents a score artifact from omitting which hierarchy,
    radius and additive energy weights produced it.
    """

    if str(stage) not in _STAGE:
        raise ValueError("pose transport stage must be coarse, medium, or fine")
    scale = float(depth_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("depth_scale must be positive")
    norm_threshold = float(zero_norm_threshold)
    if not np.isfinite(norm_threshold) or norm_threshold <= 0.0:
        raise ValueError("zero_norm_threshold must be positive")
    config = _STAGE[str(stage)]
    payload = {
        "transport_semantics": SPARSE_TRANSPORT_SEMANTICS,
        "stage": str(stage),
        "radius": int(config["radius"]),
        "relations": sorted(str(value) for value in config["relations"]),
        "depth_semantics": str(config["depth"]),
        "additive_energy_weights": [float(value) for value in config["weights"]],
        "source_bias": -0.5,
        "unmatched_sink_logit": float(config["sink"]),
        "depth_scale": scale,
        "zero_norm_threshold": norm_threshold,
        "modality_similarity_floor": 0.0,
        "query_reliability": "child_mass_entropy_background_v1",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CandidatePoseFieldObservation:
    child_rows: np.ndarray
    child_weights: np.ndarray
    pose_codes: np.ndarray
    normals: np.ndarray
    relative_depth: np.ndarray
    boundary: np.ndarray
    valid: np.ndarray
    field_semantics: str = POSE_FIELD_SEMANTICS


@dataclass(frozen=True)
class CandidateConditionedPoseAttribution:
    child_rows: np.ndarray
    child_probabilities: np.ndarray
    unmatched_probability: np.ndarray
    token_matched_probability: np.ndarray
    token_score: np.ndarray
    combined_score: float
    field_semantics: str
    production_eligible: bool


def _unit(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array, dtype=np.float64)
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-8)


def _unique_nonnegative_rows(rows: np.ndarray, *, name: str) -> None:
    for token, values in enumerate(np.asarray(rows, dtype=np.int64)):
        valid = values[values >= 0]
        if np.unique(valid).size != valid.size:
            raise ValueError(f"{name} contains a duplicate child ID at token {token}")


def _validate_confidence(value: np.ndarray, shape: tuple[int, ...], *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if (
        result.shape != shape or np.any(~np.isfinite(result))
        or np.any(result < 0.0) or np.any(result > 1.0)
    ):
        raise ValueError(f"invalid {name}")
    return result


def sparse_candidate_conditioned_pose_transport(
    retrieval: PureRadioPhysicalRetrieval,
    query: QueryPoseTransportObservation,
    candidate: CandidatePoseTransportObservation,
    hierarchy: PoseTransportHierarchy,
    *,
    stage: str,
    depth_scale: float = 0.25,
    zero_norm_threshold: float = 1e-8,
) -> SparsePoseTransportResult:
    """Locally reattribute retrieval mass with a source-substochastic softmax.

    Candidate edges are restricted by token radius and physical hierarchy.
    Every source has an explicit unmatched sink, so its outgoing matched mass
    never exceeds ``q_ret``.  This NumPy implementation is the deterministic
    reference for a future trainable/GPU path, not a production model.
    """

    if str(stage) not in _STAGE:
        raise ValueError("pose transport stage must be coarse, medium, or fine")
    config = _STAGE[str(stage)]
    _sha256(query.radio_content_sha256, name="query radio content")
    _sha256(query.readout_content_sha256, name="query readout content")
    _sha256(candidate.pose_field_content_sha256, name="map pose field content")
    _sha256(hierarchy.content_sha256, name="transport hierarchy content")
    if not str(query.query_id) or not str(candidate.basin_id):
        raise ValueError("query and basin identity must be nonempty")
    if query.normal_frame != "camera" or candidate.normal_frame != "camera":
        raise ValueError("query and map normals must both be in camera frame")
    if query.depth_semantics != config["depth"] or candidate.depth_semantics != config["depth"]:
        raise ValueError("relative-depth semantics differ from transport stage")
    if not np.isfinite(depth_scale) or float(depth_scale) <= 0.0:
        raise ValueError("depth_scale must be positive")
    if not np.isfinite(zero_norm_threshold) or float(zero_norm_threshold) <= 0.0:
        raise ValueError("zero_norm_threshold must be positive")

    source_rows = np.asarray(retrieval.token_child_rows, dtype=np.int64)
    source_mass = np.asarray(retrieval.token_child_probabilities, dtype=np.float64)
    token_xy = np.asarray(retrieval.token_xy, dtype=np.int64)
    target_rows = np.asarray(candidate.child_rows, dtype=np.int64)
    target_mass = np.asarray(candidate.child_weights, dtype=np.float64)
    if (
        source_rows.ndim != 2 or source_mass.shape != source_rows.shape
        or target_rows.ndim != 2 or target_mass.shape != target_rows.shape
        or target_rows.shape[0] != source_rows.shape[0]
        or token_xy.shape != (source_rows.shape[0], 2)
        or np.any(~np.isfinite(source_mass)) or np.any(source_mass < 0.0)
        or np.any(~np.isfinite(target_mass)) or np.any(target_mass < 0.0)
    ):
        raise ValueError("invalid source/target transport mass")
    _unique_nonnegative_rows(source_rows, name="query")
    _unique_nonnegative_rows(target_rows, name="candidate")
    if (
        np.any(np.sum(source_mass, axis=1) > 1.0 + 2e-5)
        or np.any(np.sum(target_mass, axis=1) > 1.0 + 2e-5)
        or np.any(source_rows < -1) or np.any(target_rows < -1)
        or np.unique(token_xy, axis=0).shape[0] != token_xy.shape[0]
    ):
        raise ValueError("invalid source/target transport mass")
    token_count, source_slots = source_rows.shape
    target_slots = target_rows.shape[1]
    child_parent = np.asarray(hierarchy.child_parent_ids, dtype=np.int64).reshape(-1)
    child_support = np.asarray(hierarchy.child_support_ids, dtype=np.int64).reshape(-1)
    child_count = child_parent.size
    offsets = np.asarray(hierarchy.adjacency_offsets, dtype=np.int64).reshape(-1)
    adjacency_rows = np.asarray(hierarchy.adjacency_child_rows, dtype=np.int64).reshape(-1)
    if (
        child_support.shape != (child_count,) or offsets.shape != (child_count + 1,)
        or offsets[0] != 0 or offsets[-1] != adjacency_rows.size
        or np.any(np.diff(offsets) < 0) or np.any(adjacency_rows < 0)
        or np.any(adjacency_rows >= child_count)
        or np.any(child_parent < 0) or np.any(child_support < -1)
        or np.any(source_rows >= child_count) or np.any(target_rows >= child_count)
    ):
        raise ValueError("invalid pose transport hierarchy")
    expected_hierarchy_sha256 = pose_transport_hierarchy_content_sha256(
        child_parent, child_support, offsets, adjacency_rows
    )
    if str(hierarchy.content_sha256) != expected_hierarchy_sha256:
        raise ValueError("pose transport hierarchy content hash differs")
    adjacency = tuple(
        tuple(adjacency_rows[offsets[row] : offsets[row + 1]].tolist())
        for row in range(child_count)
    )
    if any(len(set(values)) != len(values) for values in adjacency):
        raise ValueError("pose transport hierarchy adjacency contains duplicates")
    if any(row in values for row, values in enumerate(adjacency)):
        raise ValueError("pose transport hierarchy adjacency contains self edges")
    adjacency_sets = tuple(set(values) for values in adjacency)
    if any(row not in adjacency_sets[neighbour] for row, values in enumerate(adjacency) for neighbour in values):
        raise ValueError("pose transport hierarchy adjacency must be symmetric")

    query_code = np.asarray(query.pose_codes, dtype=np.float64)
    query_normal = np.asarray(query.normals_camera, dtype=np.float64)
    query_depth = np.asarray(query.relative_depth, dtype=np.float64).reshape(-1)
    query_boundary = np.asarray(query.boundary, dtype=np.float64).reshape(-1)
    map_code = np.asarray(candidate.pose_codes, dtype=np.float64)
    map_normal = np.asarray(candidate.normals_camera, dtype=np.float64)
    map_depth = np.asarray(candidate.relative_depth, dtype=np.float64)
    map_boundary = np.asarray(candidate.boundary, dtype=np.float64)
    double_sided = np.asarray(candidate.double_sided, dtype=bool)
    if query_code.ndim != 2 or not 2 <= query_code.shape[1] <= 64:
        raise ValueError("query pose code dimension must lie in [2,64]")
    dimension = query_code.shape[1]
    if (
        query_code.shape != (token_count, dimension)
        or query_normal.shape != (token_count, 3)
        or query_depth.shape != (token_count,) or query_boundary.shape != (token_count,)
        or map_code.shape != (token_count, target_slots, dimension)
        or map_normal.shape != (token_count, target_slots, 3)
        or map_depth.shape != (token_count, target_slots)
        or map_boundary.shape != (token_count, target_slots)
        or double_sided.shape != (token_count, target_slots)
        or any(np.any(~np.isfinite(value)) for value in (
            query_code, query_normal, query_depth, query_boundary,
            map_code, map_normal, map_depth, map_boundary,
        ))
        or np.any((query_boundary < 0.0) | (query_boundary > 1.0))
        or np.any((map_boundary < 0.0) | (map_boundary > 1.0))
    ):
        raise ValueError("invalid multimodal pose transport observation")

    q_shapes = (token_count,)
    m_shapes = (token_count, target_slots)
    q_valid = {
        "feature": np.asarray(query.pose_code_valid, dtype=bool).reshape(-1),
        "normal": np.asarray(query.normal_valid, dtype=bool).reshape(-1),
        "depth": np.asarray(query.depth_valid, dtype=bool).reshape(-1),
        "boundary": np.asarray(query.boundary_valid, dtype=bool).reshape(-1),
    }
    m_valid = {
        "feature": np.asarray(candidate.pose_code_valid, dtype=bool),
        "normal": np.asarray(candidate.normal_valid, dtype=bool),
        "depth": np.asarray(candidate.depth_valid, dtype=bool),
        "boundary": np.asarray(candidate.boundary_valid, dtype=bool),
    }
    if any(value.shape != q_shapes for value in q_valid.values()) or any(
        value.shape != m_shapes for value in m_valid.values()
    ):
        raise ValueError("modality validity arrays differ")
    q_conf = {
        "feature": _validate_confidence(query.pose_code_confidence, q_shapes, name="query pose-code confidence"),
        "normal": _validate_confidence(query.normal_confidence, q_shapes, name="query normal confidence"),
        "depth": _validate_confidence(query.depth_confidence, q_shapes, name="query depth confidence"),
        "boundary": _validate_confidence(query.boundary_confidence, q_shapes, name="query boundary confidence"),
    }
    m_conf = {
        "feature": _validate_confidence(candidate.pose_code_confidence, m_shapes, name="map pose-code confidence"),
        "normal": _validate_confidence(candidate.normal_confidence, m_shapes, name="map normal confidence"),
        "depth": _validate_confidence(candidate.depth_confidence, m_shapes, name="map depth confidence"),
        "boundary": _validate_confidence(candidate.boundary_confidence, m_shapes, name="map boundary confidence"),
    }
    q_code_norm = np.linalg.norm(query_code, axis=1)
    q_normal_norm = np.linalg.norm(query_normal, axis=1)
    m_code_norm = np.linalg.norm(map_code, axis=2)
    m_normal_norm = np.linalg.norm(map_normal, axis=2)
    q_valid["feature"] &= q_code_norm >= float(zero_norm_threshold)
    q_valid["normal"] &= q_normal_norm >= float(zero_norm_threshold)
    m_valid["feature"] &= m_code_norm >= float(zero_norm_threshold)
    m_valid["normal"] &= m_normal_norm >= float(zero_norm_threshold)
    query_code = _unit(query_code)
    query_normal = _unit(query_normal)
    map_code = _unit(map_code)
    map_normal = _unit(map_normal)

    matched_source = np.zeros_like(source_mass, dtype=np.float64)
    unmatched_source = source_mass.copy()
    edge_count = 0
    radius = int(config["radius"])
    beta_feature, beta_normal, beta_depth, beta_boundary, beta_hierarchy, beta_layout = config["weights"]
    for token in range(token_count):
        delta = np.max(np.abs(token_xy - token_xy[token]), axis=1)
        local_tokens = np.flatnonzero(delta <= radius)
        for source_slot in range(source_slots):
            child_q = int(source_rows[token, source_slot])
            mass_q = float(source_mass[token, source_slot])
            if child_q < 0 or mass_q <= 0.0:
                continue
            logits = []
            for target_token in local_tokens.tolist():
                layout_distance = float(np.max(np.abs(token_xy[target_token] - token_xy[token])))
                for target_slot in range(target_slots):
                    child_m = int(target_rows[target_token, target_slot])
                    mass_m = float(target_mass[target_token, target_slot])
                    if child_m < 0 or mass_m <= 0.0:
                        continue
                    if child_q == child_m:
                        relation, hierarchy_score = "exact", 1.0
                    elif child_support[child_q] >= 0 and child_support[child_q] == child_support[child_m]:
                        relation, hierarchy_score = "support", 0.7
                    elif child_m in adjacency_sets[child_q]:
                        relation, hierarchy_score = "adjacent", 0.45
                    elif child_parent[child_q] == child_parent[child_m]:
                        relation, hierarchy_score = "parent", 0.25
                    else:
                        continue
                    if relation not in config["relations"]:
                        continue
                    logit = -0.5 + np.log(max(mass_m, 1e-12))
                    if q_valid["feature"][token] and m_valid["feature"][target_token, target_slot]:
                        confidence = np.sqrt(q_conf["feature"][token] * m_conf["feature"][target_token, target_slot])
                        similarity = 0.5 * (
                            1.0 + float(np.dot(query_code[token], map_code[target_token, target_slot]))
                        )
                        logit += beta_feature * confidence * float(np.clip(similarity, 0.0, 1.0))
                    if q_valid["normal"][token] and m_valid["normal"][target_token, target_slot]:
                        cosine = float(np.dot(query_normal[token], map_normal[target_token, target_slot]))
                        cosine = abs(cosine) if double_sided[target_token, target_slot] else cosine
                        confidence = np.sqrt(q_conf["normal"][token] * m_conf["normal"][target_token, target_slot])
                        similarity = cosine if double_sided[target_token, target_slot] else 0.5 * (1.0 + cosine)
                        logit += beta_normal * confidence * float(np.clip(similarity, 0.0, 1.0))
                    if beta_depth and q_valid["depth"][token] and m_valid["depth"][target_token, target_slot]:
                        similarity = 1.0 - min(
                            abs(query_depth[token] - map_depth[target_token, target_slot])
                            / float(depth_scale),
                            1.0,
                        )
                        confidence = np.sqrt(q_conf["depth"][token] * m_conf["depth"][target_token, target_slot])
                        logit += beta_depth * confidence * similarity
                    if beta_boundary and q_valid["boundary"][token] and m_valid["boundary"][target_token, target_slot]:
                        similarity = 1.0 - abs(
                            query_boundary[token] - map_boundary[target_token, target_slot]
                        )
                        confidence = np.sqrt(q_conf["boundary"][token] * m_conf["boundary"][target_token, target_slot])
                        logit += beta_boundary * confidence * similarity
                    layout_score = 1.0 - layout_distance / float(radius + 1)
                    logit += beta_hierarchy * hierarchy_score + beta_layout * layout_score
                    logits.append(logit)
            if logits:
                maximum = max(float(config["sink"]), max(logits))
                sink = np.exp(float(config["sink"]) - maximum)
                edge = np.exp(np.asarray(logits, dtype=np.float64) - maximum)
                denominator = sink + float(np.sum(edge))
                matched_source[token, source_slot] = mass_q * float(np.sum(edge)) / denominator
                unmatched_source[token, source_slot] = mass_q * sink / denominator
                edge_count += len(logits)
    if np.max(np.abs(matched_source + unmatched_source - source_mass), initial=0.0) > 2e-7:
        raise AssertionError("source-wise pose transport does not conserve retrieval mass")
    token_matched = np.sum(matched_source, axis=1)
    token_score = 2.0 * token_matched - 1.0
    reliability = query_only_pose_reliability_weights(retrieval)
    denominator = float(np.sum(reliability))
    combined = -1.0 if denominator <= 1e-12 else float(np.sum(reliability * token_score) / denominator)
    return SparsePoseTransportResult(
        source_child_rows=source_rows.copy(),
        source_child_probabilities=source_mass.astype(np.float32),
        matched_source_probability=matched_source.astype(np.float32),
        unmatched_source_probability=unmatched_source.astype(np.float32),
        token_matched_probability=token_matched.astype(np.float32),
        token_score=token_score.astype(np.float32), combined_score=combined,
        edge_count=int(edge_count), stage=str(stage),
        transport_semantics=SPARSE_TRANSPORT_SEMANTICS,
        query_id=str(query.query_id), basin_id=str(candidate.basin_id),
        query_radio_content_sha256=str(query.radio_content_sha256),
        query_readout_content_sha256=str(query.readout_content_sha256),
        map_pose_field_content_sha256=str(candidate.pose_field_content_sha256),
        hierarchy_content_sha256=str(hierarchy.content_sha256),
        transport_model_content_sha256=reference_pose_transport_model_content_sha256(
            str(stage), depth_scale=float(depth_scale),
            zero_norm_threshold=float(zero_norm_threshold),
        ),
        production_eligible=False,
    )


def candidate_conditioned_pose_attribution(
    retrieval: PureRadioPhysicalRetrieval,
    query_pose_codes: np.ndarray,
    query_normals: np.ndarray,
    query_relative_depth: np.ndarray,
    query_boundary: np.ndarray,
    query_valid: np.ndarray,
    candidate: CandidatePoseFieldObservation,
    *,
    depth_scale: float = 0.25,
    feature_power: float = 1.0,
    normal_power: float = 0.5,
    depth_power: float = 0.5,
    boundary_power: float = 0.25,
) -> CandidateConditionedPoseAttribution:
    """Convert retrieval prior into a conservative basin-conditioned posterior.

    All compatibility factors lie in ``[0,1]``.  For query child ``c`` the
    matched probability is ``q_ret(c) * sum_l map_mass(l) * compatibility(l)``
    over slots with the same child.  The remainder of unit mass is unmatched.
    Thus removing map support or invalidating either side cannot improve the
    token score ``2 * matched - 1``.
    """

    if str(candidate.field_semantics) != POSE_FIELD_SEMANTICS:
        raise ValueError("candidate observation is not the dedicated pose-equivariant field")
    rows = np.asarray(retrieval.token_child_rows, dtype=np.int64)
    prior = np.asarray(retrieval.token_child_probabilities, dtype=np.float64)
    token_count, query_slots = rows.shape
    query_code = np.asarray(query_pose_codes, dtype=np.float64)
    query_normal = np.asarray(query_normals, dtype=np.float64)
    query_depth = np.asarray(query_relative_depth, dtype=np.float64).reshape(-1)
    query_edge = np.asarray(query_boundary, dtype=np.float64).reshape(-1)
    query_mask = np.asarray(query_valid, dtype=bool).reshape(-1)
    map_rows = np.asarray(candidate.child_rows, dtype=np.int64)
    map_mass = np.asarray(candidate.child_weights, dtype=np.float64)
    map_code = np.asarray(candidate.pose_codes, dtype=np.float64)
    map_normal = np.asarray(candidate.normals, dtype=np.float64)
    map_depth = np.asarray(candidate.relative_depth, dtype=np.float64)
    map_edge = np.asarray(candidate.boundary, dtype=np.float64)
    map_valid = np.asarray(candidate.valid, dtype=bool)
    if query_code.ndim != 2:
        raise ValueError("query pose code must have shape [token,dimension]")
    dimension = int(query_code.shape[1])
    if (
        prior.shape != rows.shape or query_code.shape[0] != token_count
        or query_normal.shape != (token_count, 3)
        or query_depth.shape != (token_count,) or query_edge.shape != (token_count,)
        or query_mask.shape != (token_count,) or map_rows.ndim != 2
        or map_rows.shape[0] != token_count or map_mass.shape != map_rows.shape
        or map_code.shape != map_rows.shape + (dimension,)
        or map_normal.shape != map_rows.shape + (3,)
        or map_depth.shape != map_rows.shape or map_edge.shape != map_rows.shape
        or map_valid.shape != map_rows.shape
    ):
        raise ValueError("candidate-conditioned pose observation arrays differ")
    powers = np.asarray(
        [feature_power, normal_power, depth_power, boundary_power], dtype=np.float64
    )
    if (
        dimension < 2 or dimension > 64 or np.any(~np.isfinite(powers))
        or np.any(powers < 0.0) or float(np.sum(powers)) <= 0.0
        or not np.isfinite(depth_scale) or float(depth_scale) <= 0.0
        or any(np.any(~np.isfinite(value)) for value in (
            prior, query_code, query_normal, query_depth, query_edge,
            map_mass, map_code, map_normal, map_depth, map_edge,
        ))
        or np.any(prior < 0.0) or np.any(map_mass < 0.0)
        or np.any(np.sum(prior, axis=1) > 1.0 + 2e-5)
        or np.any(np.sum(map_mass, axis=1) > 1.0 + 2e-5)
        or np.any((query_edge < 0.0) | (query_edge > 1.0))
        or np.any((map_edge < 0.0) | (map_edge > 1.0))
    ):
        raise ValueError("invalid candidate-conditioned pose evidence")
    q_code = _unit(query_code)
    m_code = _unit(map_code)
    q_normal = _unit(query_normal)
    m_normal = _unit(map_normal)
    feature = np.clip(
        (np.einsum("td,tld->tl", q_code, m_code) + 1.0) * 0.5, 0.0, 1.0
    )
    normal = np.clip(
        (np.einsum("td,tld->tl", q_normal, m_normal) + 1.0) * 0.5, 0.0, 1.0
    )
    depth = np.exp(-np.abs(query_depth[:, None] - map_depth) / float(depth_scale))
    boundary = np.clip(1.0 - np.abs(query_edge[:, None] - map_edge), 0.0, 1.0)
    components = np.stack([feature, normal, depth, boundary], axis=-1)
    compatibility = np.exp(
        np.sum(powers * np.log(np.maximum(components, 1e-12)), axis=-1)
        / float(np.sum(powers))
    )
    compatibility *= query_mask[:, None] * map_valid
    match = rows[:, :, None] == map_rows[:, None, :]
    match &= (rows[:, :, None] >= 0) & (map_rows[:, None, :] >= 0)
    transferred = prior[:, :, None] * map_mass[:, None, :] * compatibility[:, None, :] * match
    child_probability = np.sum(transferred, axis=2)
    matched = np.clip(np.sum(child_probability, axis=1), 0.0, 1.0)
    unmatched = np.clip(1.0 - matched, 0.0, 1.0)
    token_score = 2.0 * matched - 1.0
    return CandidateConditionedPoseAttribution(
        child_rows=rows.copy(),
        child_probabilities=child_probability.astype(np.float32),
        unmatched_probability=unmatched.astype(np.float32),
        token_matched_probability=matched.astype(np.float32),
        token_score=token_score.astype(np.float32),
        combined_score=float(np.mean(token_score)),
        field_semantics=POSE_FIELD_SEMANTICS,
        # This is the frozen mathematical/reference seam.  A trained OOF
        # query readout and map pose-field artifact are still required.
        production_eligible=False,
    )
