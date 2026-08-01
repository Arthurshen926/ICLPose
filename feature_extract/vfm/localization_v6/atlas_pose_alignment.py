"""Pose-conditioned maplet-atlas rendering and continuous VFM alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization_v6.atlas_renderer import (
    atlas_scene_geometry_points,
    render_scene_depth_from_points,
    render_selected_maplet_atlases_fast,
)
from feature_extract.vfm.localization_v6.heldout_verifier import (
    accept_pose_update_bayes_factor,
    paired_chart_log_bayes_factor_gains,
    split_fit_heldout_maplets,
    zero_displacement_log_bayes_factor,
)
from feature_extract.vfm.localization_v6.local_correlation import (
    CorrelationDistribution,
    LocalCorrelationQueryCache,
    build_local_correlation_query_cache,
    local_correlation_distribution,
)
from feature_extract.vfm.localization_v6.maplet_atlas import (
    MapletFeatureAtlasBank,
)
from feature_extract.vfm.localization_v6.se3_update import (
    SE3UpdateResult,
    se3_exp,
    solve_correlation_se3_hypotheses,
)


@dataclass(frozen=True)
class AtlasAlignmentLevel:
    name: str
    feature_stride: int
    correlation_radius: int
    maximum_translation_step_m: float
    maximum_rotation_step_deg: float
    maximum_points: int = 4096
    translation_search_step_m: float | None = None
    direct_rotation_step_deg: float | None = None


@dataclass(frozen=True)
class AtlasAlignmentStep:
    level: str
    phase: str
    rendered_point_count: int
    fit_maplet_count: int
    heldout_maplet_count: int
    update_hypothesis_count: int
    heldout_update_hypothesis_count: int
    heldout_update_consistent: bool
    normalized_update_disagreement: float
    accepted: bool
    selected_hypothesis_index: int | None
    proposal_source: str | None
    verification_method: str
    delta_se3: tuple[float, ...]
    delta_translation_m: float
    delta_rotation_deg: float
    fit_before: float
    fit_after: float
    heldout_before: float
    heldout_after: float
    fit_verified_surface_count: int
    heldout_verified_surface_count: int
    used_point_count: int
    condition_number: float
    proposal_audit: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class AtlasPoseAlignmentResult:
    initial_pose_w2c: np.ndarray
    refined_pose_w2c: np.ndarray
    score: float
    fit_score: float
    heldout_score: float
    accepted_step_count: int
    steps: tuple[AtlasAlignmentStep, ...]
    selected_chart_ids: tuple[int, ...]


def _translation_search_step(level: AtlasAlignmentLevel) -> float:
    """Return the common metric radius for analytic and direct proposals."""

    return float(
        level.maximum_translation_step_m
        if level.translation_search_step_m is None
        else level.translation_search_step_m
    )


def _axis_rotation_hypotheses(
    pose_w2c: np.ndarray,
    level: AtlasAlignmentLevel,
) -> tuple[SE3UpdateResult, ...]:
    """Deterministic camera-frame rotations for direct atlas evidence."""

    step_deg = float(
        level.maximum_rotation_step_deg
        if level.direct_rotation_step_deg is None
        else level.direct_rotation_step_deg
    )
    if step_deg <= 0.0:
        return ()
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    results = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            delta = np.zeros((6,), dtype=np.float64)
            delta[axis] = sign * np.deg2rad(step_deg)
            results.append(
                SE3UpdateResult(
                    delta=delta,
                    updated_pose_w2c=se3_exp(delta) @ pose,
                    covariance=np.full((6, 6), np.inf, dtype=np.float64),
                    used_point_count=0,
                    used_maplet_count=0,
                    condition_number=float("inf"),
                    residual_rms_px=float("inf"),
                    success=True,
                )
            )
    return tuple(results)


def _component_update(
    update: SE3UpdateResult,
    pose_w2c: np.ndarray,
    *,
    component: str,
) -> SE3UpdateResult | None:
    """Project an analytic coupled update onto one SE(3) subspace."""

    source = np.asarray(update.delta, dtype=np.float64).reshape(6)
    delta = np.zeros((6,), dtype=np.float64)
    if component == "rotation":
        delta[:3] = source[:3]
        minimum = np.deg2rad(0.01)
    elif component == "translation":
        delta[3:] = source[3:]
        minimum = 1e-4
    else:
        raise ValueError("unknown SE(3) update component")
    if float(np.linalg.norm(delta)) <= float(minimum):
        return None
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    return SE3UpdateResult(
        delta=delta,
        updated_pose_w2c=se3_exp(delta) @ pose,
        covariance=np.asarray(update.covariance, dtype=np.float64),
        used_point_count=int(update.used_point_count),
        used_maplet_count=int(update.used_maplet_count),
        condition_number=float(update.condition_number),
        residual_rms_px=float(update.residual_rms_px),
        success=bool(update.success),
    )


def _joint_update_candidates(
    updates: Sequence[SE3UpdateResult],
    pose_w2c: np.ndarray,
    *,
    scales: Sequence[float] = (0.5, 1.0),
) -> tuple[tuple[SE3UpdateResult, ...], tuple[str, ...]]:
    """Retain coupled SE(3) flow updates under a finite trust-region line search.

    Rotation and translation image motion are strongly coupled on planar
    facades. Projecting the normal-equation solution onto either subspace can
    make both coordinate steps decrease likelihood even when their joint step
    points toward the correct pose. Scaled joint candidates preserve that
    geometry; fit charts select one scale and held-out charts only verify it.
    """

    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    candidates = []
    sources = []
    for update_index, update in enumerate(updates):
        source = np.asarray(update.delta, dtype=np.float64).reshape(6)
        for scale in scales:
            resolved_scale = float(scale)
            if not 0.0 < resolved_scale <= 1.0:
                raise ValueError("joint SE(3) proposal scales must be in (0, 1]")
            delta = resolved_scale * source
            if (
                float(np.linalg.norm(delta[:3])) <= np.deg2rad(0.01)
                and float(np.linalg.norm(delta[3:])) <= 1e-4
            ):
                continue
            candidates.append(
                SE3UpdateResult(
                    delta=delta,
                    updated_pose_w2c=se3_exp(delta) @ pose,
                    covariance=(
                        resolved_scale * resolved_scale
                        * np.asarray(update.covariance, dtype=np.float64)
                    ),
                    used_point_count=int(update.used_point_count),
                    used_maplet_count=int(update.used_maplet_count),
                    condition_number=float(update.condition_number),
                    residual_rms_px=float(update.residual_rms_px),
                    success=bool(update.success),
                )
            )
            sources.append(
                f"analytic_joint_{update_index}_scale_{resolved_scale:g}"
            )
    return tuple(candidates), tuple(sources)


def build_radio_final_feature_pyramid(
    feature: np.ndarray,
    *,
    base_stride: int = 16,
    strides: Sequence[int] = (16, 8, 4),
) -> dict[str, np.ndarray]:
    """Interpolate one RADIO-final map on pixel-centre aligned grids.

    This adds no RADIO intermediate activation and no new image evidence. It
    exposes the continuous, smooth final-layer field at stride 8/4 so atlas
    rendering and the analytic SE(3) solver are not quantized to 16 image
    pixels. ``align_corners=False`` uses the same half-pixel convention as the
    atlas renderer's camera-to-feature-grid scaling.
    """

    source = np.asarray(feature, dtype=np.float32)
    if source.ndim != 3 or not np.all(np.isfinite(source)):
        raise ValueError("RADIO-final feature must have shape (C,H,W)")
    resolved_base = int(base_stride)
    if resolved_base <= 0:
        raise ValueError("base feature stride must be positive")
    result: dict[str, np.ndarray] = {}
    tensor = torch.from_numpy(source)[None]
    names = {16: "coarse", 8: "middle", 4: "fine"}
    for value in strides:
        stride = int(value)
        if (
            stride <= 0
            or resolved_base % stride != 0
            or stride not in names
        ):
            raise ValueError("feature-pyramid stride is unsupported")
        factor = resolved_base // stride
        if factor == 1:
            level = tensor
        else:
            level = F.interpolate(
                tensor,
                size=(
                    int(source.shape[1]) * factor,
                    int(source.shape[2]) * factor,
                ),
                mode="bilinear",
                align_corners=False,
            )
        level = F.normalize(level, p=2, dim=1, eps=1e-8)
        result[names[stride]] = (
            level[0].detach().cpu().numpy().astype(np.float32)
        )
    return result


def _finite_evidence_score(
    correlation: CorrelationDistribution,
    fit_ids: np.ndarray,
    heldout_ids: np.ndarray,
) -> tuple[float, float, float]:
    fit = zero_displacement_log_bayes_factor(correlation, fit_ids)
    heldout = zero_displacement_log_bayes_factor(
        correlation,
        heldout_ids if heldout_ids.size else fit_ids,
    )
    if np.isfinite(fit) and np.isfinite(heldout):
        score = float(heldout + 0.25 * fit)
    else:
        score = float("-inf")
    return score, float(fit), float(heldout)


def _correlate(
    atlas: MapletFeatureAtlasBank,
    chart_ids: np.ndarray,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
    query_feature: np.ndarray,
    query_matchability: np.ndarray | None,
    level: AtlasAlignmentLevel,
    *,
    device: str,
    full_scene_depth: np.ndarray | None,
    scene_geometry_points: np.ndarray | None = None,
    query_cache: LocalCorrelationQueryCache | None = None,
) -> CorrelationDistribution:
    feature = np.asarray(query_feature, dtype=np.float32)
    scene_depth = full_scene_depth
    if scene_depth is None and scene_geometry_points is not None:
        scene_depth = render_scene_depth_from_points(
            scene_geometry_points,
            pose_w2c,
            camera,
            width=int(feature.shape[2]),
            height=int(feature.shape[1]),
        )
    rendered = render_selected_maplet_atlases_fast(
        atlas,
        chart_ids,
        pose_w2c,
        camera,
        width=int(feature.shape[2]),
        height=int(feature.shape[1]),
        full_scene_depth=scene_depth,
    )
    return local_correlation_distribution(
        rendered,
        feature,
        radius=int(level.correlation_radius),
        query_matchability=query_matchability,
        maximum_points=int(level.maximum_points),
        device=str(device),
        query_cache=query_cache,
    )


def _update_disagreement(
    first: object,
    second: object,
    level: AtlasAlignmentLevel,
) -> float:
    """Compare two independently estimated updates in physical units."""

    first_delta = np.asarray(first.delta, dtype=np.float64).reshape(6)
    second_delta = np.asarray(second.delta, dtype=np.float64).reshape(6)
    rotation_scale = max(
        np.deg2rad(float(level.maximum_rotation_step_deg)), 1e-6
    )
    translation_scale = max(
        _translation_search_step(level), 1e-6
    )
    rotation = np.linalg.norm(
        first_delta[:3] - second_delta[:3]
    ) / rotation_scale
    translation = np.linalg.norm(
        first_delta[3:] - second_delta[3:]
    ) / translation_scale
    return float(np.sqrt(0.5 * (rotation * rotation + translation * translation)))


def _heldout_update_agreement(
    update: object,
    heldout_updates: Sequence[object],
    level: AtlasAlignmentLevel,
    *,
    maximum_disagreement: float,
) -> tuple[bool, float]:
    """Require maplet-disjoint flow to predict the same SE(3) direction."""

    if not heldout_updates:
        return False, float("inf")
    first = np.asarray(update.delta, dtype=np.float64).reshape(6)
    candidates = []
    for heldout in heldout_updates:
        second = np.asarray(heldout.delta, dtype=np.float64).reshape(6)
        disagreement = _update_disagreement(update, heldout, level)
        translation_norms = (
            float(np.linalg.norm(first[3:])),
            float(np.linalg.norm(second[3:])),
        )
        rotation_norms = (
            float(np.linalg.norm(first[:3])),
            float(np.linalg.norm(second[:3])),
        )
        translation_cosine = (
            float(
                first[3:] @ second[3:]
                / max(translation_norms[0] * translation_norms[1], 1e-12)
            )
            if min(translation_norms) > 1e-4
            else 1.0
        )
        rotation_cosine = (
            float(
                first[:3] @ second[:3]
                / max(rotation_norms[0] * rotation_norms[1], 1e-12)
            )
            if min(rotation_norms) > np.deg2rad(0.02)
            else 1.0
        )
        consistent = bool(
            disagreement <= float(maximum_disagreement)
            and translation_cosine >= 0.25
            and rotation_cosine >= 0.0
        )
        candidates.append((consistent, disagreement))
    candidates.sort(key=lambda value: (not value[0], value[1]))
    return bool(candidates[0][0]), float(candidates[0][1])


def _direct_rotation_consensus_eligible(
    *,
    fit_gain: float,
    fixed_chart_gain: float,
    dynamic_all_chart_gain: float,
    heldout_gain: float,
    positive_chart_fraction: float,
    fixed_chart_count: int,
    total_chart_count: int,
    minimum_gain: float,
) -> bool:
    """Validate a bounded direct rotation by independent chart consensus.

    Direct coordinate probes are not estimated from a fit subset, so forcing
    an arbitrary four-chart split to veto the other charts is unnecessary and
    unstable.  They may instead optimize the actual atlas likelihood, but
    only when the fit subset improves and at least four fixed-surface chart
    identities, the held-out subset and the ordinary all-chart score agree.
    """

    required_charts = min(4, max(int(total_chart_count), 1))
    positive_chart_count = float(positive_chart_fraction) * int(
        fixed_chart_count
    )
    return bool(
        np.isfinite(fit_gain)
        and np.isfinite(fixed_chart_gain)
        and np.isfinite(dynamic_all_chart_gain)
        and float(fit_gain) >= float(minimum_gain)
        and float(fixed_chart_gain) >= float(minimum_gain)
        and float(dynamic_all_chart_gain) > 0.0
        and float(heldout_gain) >= 0.0
        and positive_chart_count >= required_charts
        and int(fixed_chart_count) >= required_charts
    )


def _translation_consensus_eligible(
    *,
    is_analytic: bool,
    heldout_update_consistent: bool,
    fixed_chart_gain: float,
    dynamic_all_chart_gain: float,
    positive_chart_fraction: float,
    fixed_chart_count: int,
    total_chart_count: int,
) -> bool:
    """Reject planar rotation/translation compensation as metric motion."""

    required_charts = max(4, int(np.ceil(0.5 * total_chart_count)))
    return bool(
        is_analytic
        and heldout_update_consistent
        and np.isfinite(fixed_chart_gain)
        and np.isfinite(dynamic_all_chart_gain)
        and float(fixed_chart_gain) > 0.0
        and float(dynamic_all_chart_gain) > 0.0
        and float(positive_chart_fraction) >= 0.5
        and int(fixed_chart_count) >= required_charts
    )


def refine_pose_with_maplet_atlases(
    atlases: Mapping[str, MapletFeatureAtlasBank],
    query_features: Mapping[str, np.ndarray],
    query_matchability: Mapping[str, np.ndarray | None],
    selected_chart_ids: np.ndarray,
    initial_pose_w2c: np.ndarray,
    camera: ColmapCamera,
    levels: Sequence[AtlasAlignmentLevel],
    *,
    device: str = "cuda",
    heldout_fraction: float = 0.25,
    minimum_fit_gain: float = 0.01,
    minimum_heldout_gain: float = 0.0,
    maximum_heldout_update_disagreement: float = 0.85,
    full_scene_depth: Mapping[str, np.ndarray | None] | None = None,
    use_atlas_scene_occlusion: bool = True,
    fit_maplet_ids: np.ndarray | None = None,
    heldout_maplet_ids: np.ndarray | None = None,
    query_caches: Mapping[
        tuple[str, int], LocalCorrelationQueryCache
    ]
    | None = None,
    maximum_committed_translation_updates: int = 2,
) -> AtlasPoseAlignmentResult:
    """Run render/correlate/update/verify without any point identities.

    The same selected maplet feature atlases are re-rendered after every pose
    update.  Fit and held-out chart identities are disjoint, and an update is
    committed only when both fixed-surface evidence sets support it.
    """

    maximum_translation_updates = int(
        maximum_committed_translation_updates
    )
    if maximum_translation_updates < 0:
        raise ValueError(
            "maximum_committed_translation_updates must be non-negative"
        )
    ids = np.unique(np.asarray(selected_chart_ids, dtype=np.int64))
    common = set(ids.tolist())
    for level in levels:
        if level.name not in atlases or level.name not in query_features:
            raise ValueError(f"missing atlas/query level: {level.name}")
        common &= set(
            int(value) for value in atlases[level.name].maplet_ids.tolist()
        )
    ids = np.asarray(
        [value for value in ids.tolist() if int(value) in common],
        dtype=np.int64,
    )
    initial = np.asarray(initial_pose_w2c, dtype=np.float64).reshape(4, 4)
    pose = initial.copy()
    if ids.size == 0:
        return AtlasPoseAlignmentResult(
            initial_pose_w2c=initial,
            refined_pose_w2c=pose,
            score=float("-inf"),
            fit_score=float("-inf"),
            heldout_score=float("-inf"),
            accepted_step_count=0,
            steps=(),
            selected_chart_ids=(),
        )
    if fit_maplet_ids is None and heldout_maplet_ids is None:
        fit_ids, heldout_ids = split_fit_heldout_maplets(
            ids, heldout_fraction=float(heldout_fraction)
        )
    elif fit_maplet_ids is None or heldout_maplet_ids is None:
        raise ValueError(
            "fit_maplet_ids and heldout_maplet_ids must be provided together"
        )
    else:
        fit_ids = np.intersect1d(
            ids,
            np.asarray(fit_maplet_ids, dtype=np.int64),
            assume_unique=False,
        )
        heldout_ids = np.intersect1d(
            ids,
            np.asarray(heldout_maplet_ids, dtype=np.int64),
            assume_unique=False,
        )
        if (
            fit_ids.size == 0
            or heldout_ids.size == 0
            or np.intersect1d(fit_ids, heldout_ids).size
        ):
            raise ValueError(
                "explicit fit/held-out maplet sets must be non-empty and "
                "disjoint"
            )
    steps = []
    accepted_count = 0
    last_correlation = None
    translation_commit_count = 0
    scene_geometry_by_level = {
        level.name: (
            atlas_scene_geometry_points(atlases[level.name])
            if bool(use_atlas_scene_occlusion)
            and (
                full_scene_depth is None
                or full_scene_depth.get(level.name) is None
            )
            else None
        )
        for level in levels
    }
    correlation_caches = (
        dict(query_caches)
        if query_caches is not None
        else {
            (str(level.name), int(level.correlation_radius)):
            build_local_correlation_query_cache(
                np.asarray(query_features[level.name], dtype=np.float32),
                radius=int(level.correlation_radius),
                background_samples=512,
                device=str(device),
            )
            for level in levels
        }
    )

    def active_query_cache() -> LocalCorrelationQueryCache:
        key = (
            str(active_level.name),
            int(active_level.correlation_radius),
        )
        if key not in correlation_caches:
            raise ValueError(f"missing local-correlation query cache: {key}")
        return correlation_caches[key]

    def _analytic_updates(
        correlation: CorrelationDistribution,
        current_pose: np.ndarray,
        factor_ids: np.ndarray,
    ) -> tuple[SE3UpdateResult, ...]:
        return solve_correlation_se3_hypotheses(
            correlation,
            current_pose,
            camera,
            fit_maplet_ids=factor_ids,
            maximum_null_probability=0.80,
            # Entropy includes every displacement cell plus null. Keep the
            # gate normalized when the local search radius changes.
            maximum_entropy=0.86
            * float(
                np.log(
                    max(int(correlation.offsets_xy.shape[0]) + 1, 2)
                )
            ),
            minimum_matchability=0.05,
            minimum_points=6,
            displacement_scale_xy=(
                float(active_level.feature_stride),
                float(active_level.feature_stride),
            ),
            maximum_translation_step_m=float(
                _translation_search_step(active_level)
            ),
            maximum_rotation_step_deg=float(
                active_level.maximum_rotation_step_deg
            ),
        )

    def _component_candidates(
        updates: Sequence[SE3UpdateResult],
        current_pose: np.ndarray,
        component: str,
    ) -> tuple[tuple[SE3UpdateResult, ...], tuple[str, ...]]:
        candidates = []
        sources = []
        for index, update in enumerate(updates):
            projected = _component_update(
                update, current_pose, component=component
            )
            if projected is None:
                continue
            candidates.append(projected)
            sources.append(f"analytic_{component}_{index}")
        return tuple(candidates), tuple(sources)

    def _run_phase(
        before: CorrelationDistribution,
        candidate_updates: Sequence[SE3UpdateResult],
        candidate_sources: Sequence[str],
        heldout_updates: Sequence[SE3UpdateResult],
        *,
        analytic_candidate_count: int,
        phase: str,
        atlas: MapletFeatureAtlasBank,
        feature: np.ndarray,
        matchability: np.ndarray | None,
        depth: np.ndarray | None,
        scene_geometry: np.ndarray | None,
    ) -> CorrelationDistribution:
        nonlocal pose, accepted_count, last_correlation
        nonlocal translation_commit_count
        verified = []
        all_before = zero_displacement_log_bayes_factor(before, ids)
        for hypothesis_index, (update, proposal_source) in enumerate(
            zip(candidate_updates, candidate_sources)
        ):
            after = _correlate(
                atlas,
                ids,
                update.updated_pose_w2c,
                camera,
                feature,
                matchability,
                active_level,
                device=str(device),
                full_scene_depth=depth,
                scene_geometry_points=scene_geometry,
                query_cache=active_query_cache(),
            )
            accepted, evidence = accept_pose_update_bayes_factor(
                before,
                after,
                fit_maplet_ids=fit_ids,
                heldout_maplet_ids=heldout_ids,
                minimum_fit_gain=max(
                    float(minimum_fit_gain),
                    # Fit charts propose and held-out charts independently
                    # verify the same fixed-surface improvement.  Requiring
                    # an additional hard-coded 0.10 nat rotation margin
                    # rejected genuine arbitrary-axis updates even when both
                    # disjoint sets improved.  The caller's explicit trust
                    # threshold is the correct criterion for rotation.
                    float(minimum_fit_gain)
                    if str(phase) in {"rotation", "joint"}
                    else 0.05,
                ),
                minimum_heldout_gain=float(minimum_heldout_gain),
            )
            if hypothesis_index < int(analytic_candidate_count):
                consistent, disagreement = _heldout_update_agreement(
                    update,
                    heldout_updates,
                    active_level,
                    maximum_disagreement=float(
                        maximum_heldout_update_disagreement
                    ),
                )
            else:
                # Direct coordinate proposals are selected on fit charts and
                # independently accepted on held-out chart likelihood. A
                # second noisy flow estimate is diagnostic only.
                consistent, disagreement = False, float("inf")
            fit_gain = (
                float(evidence["fit_after"] - evidence["fit_before"])
                if np.isfinite(evidence["fit_before"])
                and np.isfinite(evidence["fit_after"])
                else float("-inf")
            )
            all_after = zero_displacement_log_bayes_factor(after, ids)
            all_gain = (
                float(all_after - all_before)
                if np.isfinite(all_before) and np.isfinite(all_after)
                else float("-inf")
            )
            fixed_chart_gains = paired_chart_log_bayes_factor_gains(
                before, after, ids
            )
            fixed_gain_values = np.asarray(
                list(fixed_chart_gains.values()), dtype=np.float64
            )
            fixed_all_gain = (
                float(np.mean(fixed_gain_values))
                if fixed_gain_values.size
                else float("-inf")
            )
            positive_chart_fraction = (
                float(np.mean(fixed_gain_values > 0.0))
                if fixed_gain_values.size
                else 0.0
            )
            verified.append(
                (
                    bool(accepted),
                    fit_gain,
                    int(hypothesis_index),
                    str(proposal_source),
                    update,
                    after,
                    evidence,
                    bool(consistent),
                    float(disagreement),
                    float(all_gain),
                    float(fixed_all_gain),
                    float(positive_chart_fraction),
                    int(fixed_gain_values.size),
                )
            )
        # Fit charts select one fixed proposal. Held-out charts only accept or
        # reject that proposal; they never choose among alternatives.
        proposal = (
            max(verified, key=lambda value: (value[1], -value[2]))
            if verified
            else None
        )
        verification_method = "fit_select_then_heldout_pair_null_evidence"
        direct_consensus_override = False
        if (
            proposal is not None
            and not bool(proposal[0])
            and str(phase) == "rotation"
            and str(proposal[3]).startswith(
                "direct_atlas_rotation_axis_"
            )
        ):
            # The fit subset has already selected ``proposal`` above.  The
            # held-out charts may accept or reject that one fixed direction;
            # they must never search the other five axes and thereby become
            # part of the optimizer.
            if _direct_rotation_consensus_eligible(
                fit_gain=float(proposal[1]),
                fixed_chart_gain=float(proposal[10]),
                dynamic_all_chart_gain=float(proposal[9]),
                heldout_gain=float(
                    proposal[6]["heldout_after"]
                    - proposal[6]["heldout_before"]
                ),
                positive_chart_fraction=float(proposal[11]),
                fixed_chart_count=int(proposal[12]),
                total_chart_count=int(ids.size),
                minimum_gain=float(minimum_fit_gain),
            ):
                direct_consensus_override = True
                verification_method = (
                    "fit_selected_direct_rotation_heldout_chart_consensus"
                )
        _score, all_fit_before, all_heldout_before = _finite_evidence_score(
            before, fit_ids, heldout_ids
        )
        if proposal is None:
            last_correlation = before
            return before
        (
            proposal_accepted,
            _proposal_fit_gain,
            proposal_index,
            proposal_source,
            proposal_update,
            proposal_after,
            proposal_evidence,
            proposal_consistent,
            proposal_disagreement,
            _proposal_all_gain,
            _proposal_fixed_all_gain,
            _proposal_positive_chart_fraction,
            _proposal_fixed_chart_count,
        ) = proposal
        proposal_accepted = bool(
            proposal_accepted or direct_consensus_override
        )
        if str(phase) == "translation":
            is_analytic_translation = bool(
                int(proposal_index) < int(analytic_candidate_count)
            )
            if is_analytic_translation:
                proposal_accepted = bool(
                    proposal_accepted
                    and _translation_consensus_eligible(
                        is_analytic=True,
                        heldout_update_consistent=bool(
                            proposal_consistent
                        ),
                        fixed_chart_gain=float(
                            _proposal_fixed_all_gain
                        ),
                        dynamic_all_chart_gain=float(_proposal_all_gain),
                        positive_chart_fraction=float(
                            _proposal_positive_chart_fraction
                        ),
                        fixed_chart_count=int(
                            _proposal_fixed_chart_count
                        ),
                        total_chart_count=int(ids.size),
                    )
                )
                verification_method = (
                    "analytic_translation_disjoint_flow_and_chart_consensus"
                )
            else:
                proposal_accepted = False
                verification_method = "nonanalytic_translation_rejected"
        if bool(proposal_accepted):
            pose = proposal_update.updated_pose_w2c
            current = proposal_after
            accepted_count += 1
            if str(phase) in {"translation", "joint"} and float(
                np.linalg.norm(proposal_update.delta[3:])
            ) > 1e-4:
                translation_commit_count += 1
        else:
            current = before
        last_correlation = current
        steps.append(
            AtlasAlignmentStep(
                level=str(active_level.name),
                phase=str(phase),
                rendered_point_count=int(before.xyz.shape[0]),
                fit_maplet_count=int(fit_ids.size),
                heldout_maplet_count=int(heldout_ids.size),
                update_hypothesis_count=len(candidate_updates),
                heldout_update_hypothesis_count=len(heldout_updates),
                heldout_update_consistent=bool(proposal_consistent),
                normalized_update_disagreement=float(
                    proposal_disagreement
                ),
                accepted=bool(proposal_accepted),
                selected_hypothesis_index=int(proposal_index),
                proposal_source=str(proposal_source),
                verification_method=str(verification_method),
                delta_se3=tuple(
                    float(value)
                    for value in np.asarray(
                        proposal_update.delta, dtype=np.float64
                    ).reshape(6)
                ),
                delta_translation_m=float(
                    np.linalg.norm(proposal_update.delta[3:])
                ),
                delta_rotation_deg=float(
                    np.degrees(np.linalg.norm(proposal_update.delta[:3]))
                ),
                fit_before=float(
                    proposal_evidence.get("fit_before", all_fit_before)
                ),
                fit_after=float(
                    proposal_evidence.get("fit_after", all_fit_before)
                ),
                heldout_before=float(
                    proposal_evidence.get(
                        "heldout_before", all_heldout_before
                    )
                ),
                heldout_after=float(
                    proposal_evidence.get(
                        "heldout_after", all_heldout_before
                    )
                ),
                fit_verified_surface_count=int(
                    proposal_evidence["fit_surface_count"]
                ),
                heldout_verified_surface_count=int(
                    proposal_evidence["heldout_surface_count"]
                ),
                used_point_count=int(proposal_update.used_point_count),
                condition_number=float(proposal_update.condition_number),
                proposal_audit=tuple(
                    {
                        "hypothesis_index": int(value[2]),
                        "proposal_source": str(value[3]),
                        "accepted": bool(value[0]),
                        "fit_gain": (
                            float(value[1])
                            if np.isfinite(value[1])
                            else None
                        ),
                        "heldout_gain": (
                            float(
                                value[6]["heldout_after"]
                                - value[6]["heldout_before"]
                            )
                            if np.isfinite(value[6]["heldout_after"])
                            and np.isfinite(value[6]["heldout_before"])
                            else None
                        ),
                        "all_chart_gain": (
                            float(value[9])
                            if np.isfinite(value[9])
                            else None
                        ),
                        "fixed_chart_all_chart_gain": (
                            float(value[10])
                            if np.isfinite(value[10])
                            else None
                        ),
                        "positive_chart_fraction": float(value[11]),
                        "fixed_chart_count": int(value[12]),
                        "delta_translation_m": float(
                            np.linalg.norm(value[4].delta[3:])
                        ),
                        "delta_rotation_deg": float(
                            np.degrees(
                                np.linalg.norm(value[4].delta[:3])
                            )
                        ),
                    }
                    for value in verified
                ),
            )
        )
        return current

    # First converge orientation across the complete feature pyramid.  An
    # interleaved translation at a still-rotating coarse level can improve
    # image flow by compensating residual angular error while moving the
    # camera centre in the wrong direction.
    for level in levels:
        active_level = level
        atlas = atlases[level.name]
        feature = np.asarray(query_features[level.name], dtype=np.float32)
        matchability = query_matchability.get(level.name)
        depth = (
            None
            if full_scene_depth is None
            else full_scene_depth.get(level.name)
        )
        scene_geometry = scene_geometry_by_level[level.name]
        before = _correlate(
            atlas,
            ids,
            pose,
            camera,
            feature,
            matchability,
            level,
            device=str(device),
            full_scene_depth=depth,
            scene_geometry_points=scene_geometry,
            query_cache=active_query_cache(),
        )
        last_correlation = before

        # Build one proposal set before looking at held-out evidence.  Running
        # joint flow, projected rotation and direct atlas probes as sequential
        # phases lets call order choose a merely positive joint update before
        # a better direct proposal is ever rendered.  Worse, falling through
        # to a later family only after held-out rejection makes held-out
        # evidence part of proposal selection.  Fit charts must choose once
        # over every orientation-capable proposal at this pose; held-out
        # charts then accept or reject that single winner.
        #
        # Coupled normal-equation proposals remain valuable on planar
        # facades, while bounded direct rotations protect against a multimodal
        # RADIO flow posterior whose per-cell mean points at the wrong mode.
        fit_analytic = _analytic_updates(before, pose, fit_ids)
        heldout_analytic = _analytic_updates(before, pose, heldout_ids)
        joint_analytic, joint_sources = _joint_update_candidates(
            fit_analytic, pose
        )
        heldout_joint, _heldout_joint_sources = _joint_update_candidates(
            heldout_analytic, pose
        )
        rotation_analytic, rotation_sources = _component_candidates(
            fit_analytic, pose, "rotation"
        )
        heldout_rotation, _heldout_rotation_sources = (
            _component_candidates(heldout_analytic, pose, "rotation")
        )
        direct_rotation = _axis_rotation_hypotheses(pose, level)
        rotation_axis_sources = tuple(
            f"direct_atlas_rotation_axis_{axis}_{sign}"
            for axis in ("x", "y", "z")
            for sign in ("negative", "positive")
        )[: len(direct_rotation)]
        translation_budget_available = bool(
            translation_commit_count < maximum_translation_updates
        )
        fit_candidates = (
            *(joint_analytic if translation_budget_available else ()),
            *rotation_analytic,
            *direct_rotation,
        )
        fit_sources = (
            *joint_sources,
            *rotation_sources,
            *rotation_axis_sources,
        )
        heldout_candidates = (
            *(heldout_joint if translation_budget_available else ()),
            *heldout_rotation,
        )
        analytic_candidate_count = (
            len(joint_analytic) if translation_budget_available else 0
        ) + len(rotation_analytic)
        before = _run_phase(
            before,
            fit_candidates,
            fit_sources,
            heldout_candidates,
            analytic_candidate_count=analytic_candidate_count,
            phase="joint",
            atlas=atlas,
            feature=feature,
            matchability=matchability,
            depth=depth,
            scene_geometry=scene_geometry,
        )

    # Only after the last rotation level may metric translation compete.
    # Permit one coarse residual and one finer residual at most.  A single
    # accepted joint update used to disable this entire metric-translation
    # pass, even when the remaining fine-level RADIO flow agreed on both the
    # fit and held-out chart identities.  Conversely, leaving the budget
    # unbounded lets repeated planar-flow updates drift in depth.  The finite
    # two-update trust region preserves coarse-to-fine correction while every
    # commit still has to pass the disjoint flow and fixed-chart consensus
    # tests in ``_run_phase``.
    for level in levels:
        if translation_commit_count >= maximum_translation_updates:
            break
        active_level = level
        atlas = atlases[level.name]
        feature = np.asarray(query_features[level.name], dtype=np.float32)
        matchability = query_matchability.get(level.name)
        depth = (
            None
            if full_scene_depth is None
            else full_scene_depth.get(level.name)
        )
        scene_geometry = scene_geometry_by_level[level.name]
        before = _correlate(
            atlas,
            ids,
            pose,
            camera,
            feature,
            matchability,
            level,
            device=str(device),
            full_scene_depth=depth,
            scene_geometry_points=scene_geometry,
            query_cache=active_query_cache(),
        )
        last_correlation = before
        fit_analytic = _analytic_updates(before, pose, fit_ids)
        heldout_analytic = _analytic_updates(before, pose, heldout_ids)
        translation_analytic, translation_sources = (
            _component_candidates(fit_analytic, pose, "translation")
        )
        heldout_translation, _heldout_translation_sources = (
            _component_candidates(
                heldout_analytic, pose, "translation"
            )
        )
        # Translation remains an inferred metric update from disjoint chart
        # flow.  A direct 3-D lattice was tested on q0/q1 and accepted no
        # independently verified update while multiplying runtime, so it is
        # deliberately absent from the deployable path.
        before = _run_phase(
            before,
            translation_analytic,
            translation_sources,
            heldout_translation,
            analytic_candidate_count=len(translation_analytic),
            phase="translation",
            atlas=atlas,
            feature=feature,
            matchability=matchability,
            depth=depth,
            scene_geometry=scene_geometry,
        )

    # Report every result on the same finest requested correlation level,
    # regardless of which earlier level accepted the one translation step.
    if levels:
        active_level = levels[-1]
        atlas = atlases[active_level.name]
        feature = np.asarray(
            query_features[active_level.name], dtype=np.float32
        )
        matchability = query_matchability.get(active_level.name)
        depth = (
            None
            if full_scene_depth is None
            else full_scene_depth.get(active_level.name)
        )
        last_correlation = _correlate(
            atlas,
            ids,
            pose,
            camera,
            feature,
            matchability,
            active_level,
            device=str(device),
            full_scene_depth=depth,
            scene_geometry_points=scene_geometry_by_level[
                active_level.name
            ],
            query_cache=active_query_cache(),
        )
    if last_correlation is None:
        score = fit_score = heldout_score = float("-inf")
    else:
        score, fit_score, heldout_score = _finite_evidence_score(
            last_correlation, fit_ids, heldout_ids
        )
    return AtlasPoseAlignmentResult(
        initial_pose_w2c=initial,
        refined_pose_w2c=pose,
        score=float(score),
        fit_score=float(fit_score),
        heldout_score=float(heldout_score),
        accepted_step_count=int(accepted_count),
        steps=tuple(steps),
        selected_chart_ids=tuple(int(value) for value in ids.tolist()),
    )
