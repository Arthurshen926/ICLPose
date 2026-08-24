"""Fixed-denominator full-token surface pose energy.

This is the appearance-preserving counterpart to sparse child transport.  A
candidate render supplies, at every RADIO token, a bounded mixture of
canonical surface descriptors.  Query descriptors are produced once by the
frozen surface mapper.  Medium-stage tolerance is implemented as a local
token maximum, never as a point correspondence or PnP solve.

For query token ``u`` and candidate token ``v``::

    E(u,v) = sum_l mass(v,l) * valid(v,l) * (cos(q_u, z_vl) + 1)
    S_u    = -1 + max_{v in N(u)} E(u,v)

The final score is a fixed query-only reliability average.  Since every term
is nonnegative and the denominator is candidate independent, removing target
mass or validity cannot improve the score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


SEMANTICS = "fixed_denominator_local_fulltoken_surface_radio_energy_v1"
PARENT_GATED_SEMANTICS = "fixed_denominator_same_token_parent_gated_surface_radio_energy_v1"
CONTRASTIVE_SEMANTICS = "fixed_denominator_contrastive_fulltoken_surface_radio_energy_v1"
PARENT_GATED_CONTRASTIVE_SEMANTICS = (
    "fixed_denominator_same_token_parent_gated_contrastive_surface_radio_energy_v1"
)
CHILD_GATED_SEMANTICS = "fixed_denominator_same_token_child_gated_surface_radio_energy_v1"
CHILD_GATED_CONTRASTIVE_SEMANTICS = (
    "fixed_denominator_same_token_child_gated_contrastive_surface_radio_energy_v1"
)
PHASE_SEMANTICS = (
    "fixed_grid_additive_slot_pair_hv_min_fulltoken_surface_phase_energy_v2"
)
SHIFT_TOLERANT_PHASE_SEMANTICS = (
    "fixed_grid_additive_slot_pair_joint_shift_hv_min_fulltoken_surface_phase_energy_v2"
)
MAXMIN_PHASE_SEMANTICS = (
    "fixed_grid_max_bottleneck_slot_pair_hv_min_fulltoken_surface_phase_energy_v3"
)
SHIFT_TOLERANT_MAXMIN_PHASE_SEMANTICS = (
    "fixed_grid_max_bottleneck_slot_pair_joint_shift_hv_min_fulltoken_surface_phase_energy_v3"
)
CONDITIONAL_PHASE_CONTROL_SEMANTICS = (
    "conditional_normalized_mixture_hv_min_fulltoken_surface_phase_control_v1"
)
CONDITIONAL_SHIFT_PHASE_CONTROL_SEMANTICS = (
    "conditional_normalized_mixture_joint_shift_hv_min_fulltoken_surface_phase_control_v1"
)


@dataclass(frozen=True)
class FullTokenSurfacePoseEnergy:
    score: torch.Tensor
    token_score: torch.Tensor
    selected_target_token: torch.Tensor
    local_radius_tokens: int
    minimum_cosine_evidence: float = -1.0
    semantics: str = SEMANTICS
    production_eligible: bool = False


@dataclass(frozen=True)
class FullTokenPhasePoseEnergy:
    score: torch.Tensor
    horizontal_score: torch.Tensor
    vertical_score: torch.Tensor
    horizontal_observability: torch.Tensor
    vertical_observability: torch.Tensor
    semantics: str = PHASE_SEMANTICS
    production_eligible: bool = False
    conditional_content_score: torch.Tensor | None = None
    mass_observability: torch.Tensor | None = None
    selected_shift_y: int = 0
    selected_shift_x: int = 0


def conditional_fulltoken_phase_pose_energy_control(
    query_descriptor: torch.Tensor,
    target_descriptor: torch.Tensor,
    target_mass: torch.Tensor,
    target_valid: torch.Tensor,
    *,
    height: int = 36,
    width: int = 64,
    gradient_threshold: float = 1.0e-4,
    maximum_shift_tokens: int = 0,
) -> FullTokenPhasePoseEnergy:
    """Legacy conditional-mixture phase score retained as an empirical control.

    This score normalizes a candidate-dependent descriptor mixture at every
    token.  Removing a target slot can therefore rotate the remaining mixture
    towards the query and improve the score.  It is useful as a ranking
    control, but it does *not* satisfy evidence-disappearance monotonicity and
    is never production eligible.

    Each informative edge contributes ``cos(delta_q, delta_r)+1`` in [0,2]
    to a fixed all-edge denominator.  The fixed denominator alone is not
    enough to make the preceding candidate-dependent normalization
    conservative.
    """

    query = torch.as_tensor(query_descriptor)
    target = torch.as_tensor(target_descriptor, device=query.device, dtype=query.dtype)
    mass = torch.as_tensor(target_mass, device=query.device, dtype=query.dtype)
    valid = torch.as_tensor(target_valid, device=query.device, dtype=torch.bool)
    if query.ndim != 2 or query.shape[0] != int(height) * int(width):
        raise ValueError("query descriptor differs from the declared token grid")
    if target.ndim != 3 or target.shape[0] != query.shape[0] or target.shape[2] != query.shape[1]:
        raise ValueError("target descriptors differ from the query grid")
    if mass.shape != target.shape[:2] or valid.shape != mass.shape:
        raise ValueError("target phase mass/validity arrays differ")
    if any(
        not torch.is_floating_point(value) or not torch.isfinite(value).all()
        for value in (query, target, mass)
    ):
        raise ValueError("phase pose inputs must be finite floating tensors")
    if torch.any(mass < 0.0) or torch.any(torch.sum(mass, dim=1) > 1.0 + 2.0e-5):
        raise ValueError("target phase mass is invalid")
    threshold = float(gradient_threshold)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("gradient_threshold must be positive")
    query_norm = torch.linalg.vector_norm(query, dim=1)
    query_unit = query / query_norm[:, None].clamp_min(1.0e-8)
    target_norm = torch.linalg.vector_norm(target, dim=2)
    slot_valid = valid & (target_norm >= 1.0e-8)
    target_unit = target / target_norm[:, :, None].clamp_min(1.0e-8)
    target_grid = torch.sum(
        target_unit * mass[:, :, None] * slot_valid[:, :, None], dim=1
    )
    render_mass = torch.sum(mass * slot_valid, dim=1)
    target_grid /= torch.linalg.vector_norm(target_grid, dim=1)[:, None].clamp_min(1.0e-8)
    q = query_unit.reshape(int(height), int(width), -1)
    r = target_grid.reshape(int(height), int(width), -1)
    present = (render_mass > 1.0e-8).reshape(int(height), int(width))

    radius = int(maximum_shift_tokens)
    if radius < 0 or radius > 4:
        raise ValueError("maximum_shift_tokens must lie in [0,4]")

    def edge_fields(vertical: bool):
        if vertical:
            q0, q1 = q[:-1], q[1:]
            r0, r1 = r[:-1], r[1:]
            render_present = present[:-1] & present[1:]
            edge_mass = torch.minimum(
                render_mass.reshape(int(height), int(width))[:-1],
                render_mass.reshape(int(height), int(width))[1:],
            )
        else:
            q0, q1 = q[:, :-1], q[:, 1:]
            r0, r1 = r[:, :-1], r[:, 1:]
            render_present = present[:, :-1] & present[:, 1:]
            edge_mass = torch.minimum(
                render_mass.reshape(int(height), int(width))[:, :-1],
                render_mass.reshape(int(height), int(width))[:, 1:],
            )
        q_delta, r_delta = q1 - q0, r1 - r0
        qn = torch.linalg.vector_norm(q_delta, dim=2)
        rn = torch.linalg.vector_norm(r_delta, dim=2)
        return (
            q_delta, r_delta, qn >= threshold,
            render_present & (rn >= threshold), edge_mass, qn, rn,
        )

    horizontal_fields = edge_fields(False)
    vertical_fields = edge_fields(True)

    def shifted_direction(fields, shift_y: int, shift_x: int):
        (
            q_delta, r_delta, query_informative, render_informative,
            edge_mass, qn, rn,
        ) = fields
        edge_h, edge_w = int(q_delta.shape[0]), int(q_delta.shape[1])
        qy0, qy1 = max(0, -shift_y), min(edge_h, edge_h - shift_y)
        qx0, qx1 = max(0, -shift_x), min(edge_w, edge_w - shift_x)
        if qy1 <= qy0 or qx1 <= qx0:
            zero = torch.zeros((), device=query.device, dtype=query.dtype)
            return -1.0 + zero, zero, zero, -1.0 + zero
        ry0, ry1 = qy0 + shift_y, qy1 + shift_y
        rx0, rx1 = qx0 + shift_x, qx1 + shift_x
        qd = q_delta[qy0:qy1, qx0:qx1]
        rd = r_delta[ry0:ry1, rx0:rx1]
        informative = (
            query_informative[qy0:qy1, qx0:qx1]
            & render_informative[ry0:ry1, rx0:rx1]
        )
        cosine = torch.sum(qd * rd, dim=2) / (
            qn[qy0:qy1, qx0:qx1] * rn[ry0:ry1, rx0:rx1]
        ).clamp_min(1.0e-8)
        evidence = torch.where(
            informative, cosine.clamp(-1.0, 1.0) + 1.0, torch.zeros_like(cosine)
        )
        count = float(edge_h * edge_w)
        informative_count = torch.sum(informative)
        conditional = torch.where(
            informative_count > 0,
            -1.0 + torch.sum(evidence) / informative_count.clamp_min(1),
            torch.full((), -1.0, device=query.device, dtype=query.dtype),
        )
        structural_mass = torch.where(
            query_informative[qy0:qy1, qx0:qx1],
            edge_mass[ry0:ry1, rx0:rx1],
            torch.zeros_like(edge_mass[ry0:ry1, rx0:rx1]),
        )
        return (
            -1.0 + torch.sum(evidence) / count,
            informative_count / count,
            torch.sum(structural_mass) / count,
            conditional,
        )

    candidates = []
    for shift_y in range(-radius, radius + 1):
        for shift_x in range(-radius, radius + 1):
            horizontal, horizontal_obs, horizontal_mass, horizontal_content = shifted_direction(
                horizontal_fields, shift_y, shift_x
            )
            vertical, vertical_obs, vertical_mass, vertical_content = shifted_direction(
                vertical_fields, shift_y, shift_x
            )
            candidates.append((torch.minimum(horizontal, vertical), horizontal, vertical,
                               horizontal_obs, vertical_obs,
                               torch.minimum(horizontal_mass, vertical_mass),
                               torch.minimum(horizontal_content, vertical_content),
                               shift_y, shift_x))
    stacked = torch.stack([value[0] for value in candidates])
    best = int(torch.argmax(stacked).item())
    (
        score, horizontal, vertical, horizontal_obs, vertical_obs,
        mass_observability, conditional_content, selected_shift_y, selected_shift_x,
    ) = candidates[best]
    return FullTokenPhasePoseEnergy(
        score=score,
        horizontal_score=horizontal,
        vertical_score=vertical,
        horizontal_observability=horizontal_obs,
        vertical_observability=vertical_obs,
        semantics=(
            CONDITIONAL_PHASE_CONTROL_SEMANTICS
            if radius == 0 else CONDITIONAL_SHIFT_PHASE_CONTROL_SEMANTICS
        ),
        conditional_content_score=conditional_content,
        mass_observability=mass_observability,
        selected_shift_y=int(selected_shift_y),
        selected_shift_x=int(selected_shift_x),
    )


def conservative_fulltoken_phase_pose_energy(
    query_descriptor: torch.Tensor,
    target_descriptor: torch.Tensor,
    target_mass: torch.Tensor,
    target_valid: torch.Tensor,
    *,
    height: int = 36,
    width: int = 64,
    gradient_threshold: float = 1.0e-4,
    maximum_shift_tokens: int = 0,
    slot_pair_reduction: str = "additive_product",
) -> FullTokenPhasePoseEnergy:
    """Strict additive full-layout phase energy without correspondences.

    For each adjacent token pair, every pair of rendered surface slots is an
    immutable atom.  Its evidence is

    ``mass_a * mass_b * valid_a * valid_b * (cos(delta_q, z_b-z_a)+1)``.

    ``additive_product`` sums all pair atoms using product mass.  The
    ``maximum_bottleneck`` variant takes the best pair using the smaller of
    its endpoint masses.  It avoids dilution by unrelated surface pairs while
    remaining conservative: deleting a pair or decreasing either endpoint
    mass cannot increase a maximum of nonnegative atoms.  In both cases the
    denominator contains every edge and no candidate-dependent descriptor
    mixture is renormalized.  The final minimum requires both image axes to
    agree instead of allowing one to compensate the other.
    """

    query = torch.as_tensor(query_descriptor)
    target = torch.as_tensor(target_descriptor, device=query.device, dtype=query.dtype)
    mass = torch.as_tensor(target_mass, device=query.device, dtype=query.dtype)
    valid = torch.as_tensor(target_valid, device=query.device, dtype=torch.bool)
    if query.ndim != 2 or query.shape[0] != int(height) * int(width):
        raise ValueError("query descriptor differs from the declared token grid")
    if target.ndim != 3 or target.shape[0] != query.shape[0] or target.shape[2] != query.shape[1]:
        raise ValueError("target descriptors differ from the query grid")
    if mass.shape != target.shape[:2] or valid.shape != mass.shape:
        raise ValueError("target phase mass/validity arrays differ")
    if any(
        not torch.is_floating_point(value) or not torch.isfinite(value).all()
        for value in (query, target, mass)
    ):
        raise ValueError("phase pose inputs must be finite floating tensors")
    if torch.any(mass < 0.0) or torch.any(torch.sum(mass, dim=1) > 1.0 + 2.0e-5):
        raise ValueError("target phase mass is invalid")
    threshold = float(gradient_threshold)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("gradient_threshold must be positive")
    reduction = str(slot_pair_reduction)
    if reduction not in ("additive_product", "maximum_bottleneck"):
        raise ValueError("unknown phase slot-pair reduction")
    query_norm = torch.linalg.vector_norm(query, dim=1)
    query_unit = query / query_norm[:, None].clamp_min(1.0e-8)
    target_norm = torch.linalg.vector_norm(target, dim=2)
    slot_valid = valid & (target_norm >= 1.0e-8)
    target_unit = target / target_norm[:, :, None].clamp_min(1.0e-8)
    q = query_unit.reshape(int(height), int(width), -1)
    slots = target_unit.reshape(int(height), int(width), target.shape[1], -1)
    slot_mass = mass.reshape(int(height), int(width), target.shape[1])
    slot_present = slot_valid.reshape(int(height), int(width), target.shape[1])

    radius = int(maximum_shift_tokens)
    if radius < 0 or radius > 4:
        raise ValueError("maximum_shift_tokens must lie in [0,4]")

    def edge_fields(vertical: bool):
        if vertical:
            q0, q1 = q[:-1], q[1:]
            r0, r1 = slots[:-1], slots[1:]
            m0, m1 = slot_mass[:-1], slot_mass[1:]
            v0, v1 = slot_present[:-1], slot_present[1:]
        else:
            q0, q1 = q[:, :-1], q[:, 1:]
            r0, r1 = slots[:, :-1], slots[:, 1:]
            m0, m1 = slot_mass[:, :-1], slot_mass[:, 1:]
            v0, v1 = slot_present[:, :-1], slot_present[:, 1:]
        q_delta = q1 - q0
        r_delta = r1[:, :, None, :, :] - r0[:, :, :, None, :]
        qn = torch.linalg.vector_norm(q_delta, dim=2)
        rn = torch.linalg.vector_norm(r_delta, dim=4)
        if reduction == "additive_product":
            pair_weight = m0[:, :, :, None] * m1[:, :, None, :]
        else:
            pair_weight = torch.minimum(
                m0[:, :, :, None], m1[:, :, None, :]
            )
        pair_weight = pair_weight * v0[:, :, :, None] * v1[:, :, None, :]
        pair_informative = rn >= threshold
        return q_delta, r_delta, qn >= threshold, pair_weight, pair_informative, qn, rn

    horizontal_fields = edge_fields(False)
    vertical_fields = edge_fields(True)

    def shifted_direction(fields, shift_y: int, shift_x: int) -> tuple[torch.Tensor, torch.Tensor]:
        q_delta, r_delta, query_informative, pair_weight, pair_informative, qn, rn = fields
        edge_h, edge_w = int(q_delta.shape[0]), int(q_delta.shape[1])
        qy0, qy1 = max(0, -shift_y), min(edge_h, edge_h - shift_y)
        qx0, qx1 = max(0, -shift_x), min(edge_w, edge_w - shift_x)
        if qy1 <= qy0 or qx1 <= qx0:
            zero = torch.zeros((), device=query.device, dtype=query.dtype)
            return -1.0 + zero, zero
        ry0, ry1 = qy0 + shift_y, qy1 + shift_y
        rx0, rx1 = qx0 + shift_x, qx1 + shift_x
        qd = q_delta[qy0:qy1, qx0:qx1]
        rd = r_delta[ry0:ry1, rx0:rx1]
        weight = pair_weight[ry0:ry1, rx0:rx1]
        informative = (
            query_informative[qy0:qy1, qx0:qx1, None, None]
            & pair_informative[ry0:ry1, rx0:rx1]
        )
        cosine = torch.sum(qd[:, :, None, None, :] * rd, dim=4) / (
            qn[qy0:qy1, qx0:qx1, None, None] * rn[ry0:ry1, rx0:rx1]
        ).clamp_min(1.0e-8)
        above_floor = torch.where(
            informative,
            weight * (cosine.clamp(-1.0, 1.0) + 1.0),
            torch.zeros_like(cosine),
        )
        count = float(edge_h * edge_w)
        observable_weight = torch.where(informative, weight, torch.zeros_like(weight))
        if reduction == "additive_product":
            evidence_sum = torch.sum(above_floor)
            observability_sum = torch.sum(observable_weight)
        else:
            evidence_sum = torch.sum(torch.amax(above_floor, dim=(2, 3)))
            observability_sum = torch.sum(torch.amax(observable_weight, dim=(2, 3)))
        return -1.0 + evidence_sum / count, observability_sum / count

    candidates = []
    for shift_y in range(-radius, radius + 1):
        for shift_x in range(-radius, radius + 1):
            horizontal, horizontal_obs = shifted_direction(
                horizontal_fields, shift_y, shift_x
            )
            vertical, vertical_obs = shifted_direction(
                vertical_fields, shift_y, shift_x
            )
            candidates.append((torch.minimum(horizontal, vertical), horizontal, vertical,
                               horizontal_obs, vertical_obs))
    stacked = torch.stack([value[0] for value in candidates])
    best = int(torch.argmax(stacked).item())
    score, horizontal, vertical, horizontal_obs, vertical_obs = candidates[best]
    return FullTokenPhasePoseEnergy(
        score=score,
        horizontal_score=horizontal,
        vertical_score=vertical,
        horizontal_observability=horizontal_obs,
        vertical_observability=vertical_obs,
        semantics=(
            (PHASE_SEMANTICS if radius == 0 else SHIFT_TOLERANT_PHASE_SEMANTICS)
            if reduction == "additive_product"
            else (
                MAXMIN_PHASE_SEMANTICS
                if radius == 0 else SHIFT_TOLERANT_MAXMIN_PHASE_SEMANTICS
            )
        ),
    )


def child_gated_fulltoken_surface_pose_energy(
    query_descriptor: torch.Tensor,
    query_reliability: torch.Tensor,
    token_xy: np.ndarray,
    query_child_ids: torch.Tensor,
    query_child_probability: torch.Tensor,
    target_descriptor: torch.Tensor,
    target_mass: torch.Tensor,
    target_valid: torch.Tensor,
    target_child_ids: torch.Tensor,
    *,
    minimum_cosine_evidence: float = -1.0,
) -> FullTokenSurfacePoseEnergy:
    """Fine physical-region version of the conservative identity gate.

    Child IDs are not hard correspondences: every query token carries a soft
    posterior over anonymous surface regions, and all target slots contribute
    proportionally.  This merely evaluates a rendered pose hypothesis.
    """

    value = parent_gated_fulltoken_surface_pose_energy(
        query_descriptor, query_reliability, token_xy,
        query_child_ids, query_child_probability,
        target_descriptor, target_mass, target_valid, target_child_ids,
        minimum_cosine_evidence=float(minimum_cosine_evidence),
    )
    return FullTokenSurfacePoseEnergy(
        score=value.score,
        token_score=value.token_score,
        selected_target_token=value.selected_target_token,
        local_radius_tokens=0,
        minimum_cosine_evidence=float(minimum_cosine_evidence),
        semantics=(
            CHILD_GATED_CONTRASTIVE_SEMANTICS
            if float(minimum_cosine_evidence) > -1.0
            else CHILD_GATED_SEMANTICS
        ),
    )


def parent_gated_fulltoken_surface_pose_energy(
    query_descriptor: torch.Tensor,
    query_reliability: torch.Tensor,
    token_xy: np.ndarray,
    query_parent_ids: torch.Tensor,
    query_parent_probability: torch.Tensor,
    target_descriptor: torch.Tensor,
    target_mass: torch.Tensor,
    target_valid: torch.Tensor,
    target_parent_ids: torch.Tensor,
    *,
    minimum_cosine_evidence: float = -1.0,
) -> FullTokenSurfacePoseEnergy:
    """Require both RADIO appearance and pose-free parent-layout evidence.

    The above-floor evidence is the product of two bounded nonnegative
    channels at the same token. Consequently deleting appearance mass,
    parent mass, or validity cannot improve the result. Parent IDs are a soft
    physical-region carrier, not point correspondences.
    """

    appearance = fulltoken_surface_pose_energy(
        query_descriptor, query_reliability, token_xy,
        target_descriptor, target_mass, target_valid,
        local_radius_tokens=0,
        minimum_cosine_evidence=float(minimum_cosine_evidence),
    )
    query = torch.as_tensor(query_descriptor)
    reliability = torch.as_tensor(query_reliability, device=query.device, dtype=query.dtype).reshape(-1)
    query_ids = torch.as_tensor(query_parent_ids, device=query.device, dtype=torch.long)
    query_probability = torch.as_tensor(
        query_parent_probability, device=query.device, dtype=query.dtype
    )
    target_ids = torch.as_tensor(target_parent_ids, device=query.device, dtype=torch.long)
    mass = torch.as_tensor(target_mass, device=query.device, dtype=query.dtype)
    valid = torch.as_tensor(target_valid, device=query.device, dtype=torch.bool)
    if (
        query_ids.ndim != 2 or query_probability.shape != query_ids.shape
        or target_ids.ndim != 2 or mass.shape != target_ids.shape
        or valid.shape != target_ids.shape or query_ids.shape[0] != target_ids.shape[0]
        or query_ids.shape[0] != query.shape[0]
    ):
        raise ValueError("parent posterior and target parent arrays differ")
    if (
        not torch.isfinite(query_probability).all()
        or torch.any(query_probability < 0.0)
        or torch.any(torch.sum(query_probability, dim=1) > 1.0 + 2.0e-5)
    ):
        raise ValueError("query parent posterior is invalid")
    parent_overlap = torch.sum(
        query_probability[:, :, None]
        * mass[:, None, :]
        * valid[:, None, :]
        * (query_ids[:, :, None] == target_ids[:, None, :]),
        dim=(1, 2),
    ).clamp(0.0, 1.0)
    appearance_evidence = appearance.token_score + 1.0
    token_score = -1.0 + appearance_evidence * parent_overlap
    denominator = torch.sum(reliability)
    score = torch.where(
        denominator > 1.0e-12,
        torch.sum(reliability * token_score) / denominator.clamp_min(1.0e-12),
        torch.full((), -1.0, device=query.device, dtype=query.dtype),
    )
    return FullTokenSurfacePoseEnergy(
        score=score,
        token_score=token_score,
        selected_target_token=appearance.selected_target_token,
        local_radius_tokens=0,
        minimum_cosine_evidence=float(minimum_cosine_evidence),
        semantics=(
            PARENT_GATED_CONTRASTIVE_SEMANTICS
            if float(minimum_cosine_evidence) > -1.0
            else PARENT_GATED_SEMANTICS
        ),
    )


def _neighbour_tokens(token_xy: np.ndarray, radius: int) -> np.ndarray:
    xy = np.asarray(token_xy, dtype=np.int64)
    if xy.ndim != 2 or xy.shape[1] != 2 or np.unique(xy, axis=0).shape[0] != xy.shape[0]:
        raise ValueError("token_xy must contain unique (x,y) rows")
    if int(radius) < 0 or int(radius) > 4:
        raise ValueError("local_radius_tokens must lie in [0,4]")
    lookup = {tuple(row.tolist()): index for index, row in enumerate(xy)}
    neighbours = np.full((xy.shape[0], (2 * int(radius) + 1) ** 2), -1, dtype=np.int64)
    for token, (x, y) in enumerate(xy.tolist()):
        column = 0
        for dy in range(-int(radius), int(radius) + 1):
            for dx in range(-int(radius), int(radius) + 1):
                neighbours[token, column] = lookup.get((x + dx, y + dy), -1)
                column += 1
    return neighbours


def fulltoken_surface_pose_energy(
    query_descriptor: torch.Tensor,
    query_reliability: torch.Tensor,
    token_xy: np.ndarray,
    target_descriptor: torch.Tensor,
    target_mass: torch.Tensor,
    target_valid: torch.Tensor,
    *,
    local_radius_tokens: int = 1,
    minimum_cosine_evidence: float = -1.0,
) -> FullTokenSurfacePoseEnergy:
    """Score one candidate without child-ID equality or hard correspondences."""

    query = torch.as_tensor(query_descriptor)
    reliability = torch.as_tensor(
        query_reliability, device=query.device, dtype=query.dtype
    ).reshape(-1)
    target = torch.as_tensor(
        target_descriptor, device=query.device, dtype=query.dtype
    )
    mass = torch.as_tensor(target_mass, device=query.device, dtype=query.dtype)
    valid = torch.as_tensor(target_valid, device=query.device, dtype=torch.bool)
    if query.ndim != 2 or target.ndim != 3 or target.shape[0] != query.shape[0]:
        raise ValueError("query/target descriptors must have shapes [T,D] and [T,L,D]")
    if target.shape[2] != query.shape[1] or mass.shape != target.shape[:2] or valid.shape != mass.shape:
        raise ValueError("target descriptor, mass and validity arrays differ")
    if reliability.shape != (query.shape[0],):
        raise ValueError("query reliability must have one value per token")
    floating = (query, reliability, target, mass)
    if any(not torch.is_floating_point(value) or not torch.isfinite(value).all() for value in floating):
        raise ValueError("full-token pose energy inputs must be finite floating tensors")
    if torch.any(reliability < 0.0) or torch.any(mass < 0.0):
        raise ValueError("full-token pose energy weights must be nonnegative")
    if torch.any(torch.sum(mass, dim=1) > 1.0 + 2.0e-5):
        raise ValueError("target surface mass exceeds one")
    cosine_floor = float(minimum_cosine_evidence)
    if not np.isfinite(cosine_floor) or not -1.0 <= cosine_floor < 1.0:
        raise ValueError("minimum_cosine_evidence must lie in [-1,1)")
    threshold = 1.0e-8
    query_norm = torch.linalg.vector_norm(query, dim=1)
    target_norm = torch.linalg.vector_norm(target, dim=2)
    valid = valid & (target_norm >= threshold)
    query_valid = query_norm >= threshold
    query_unit = query / query_norm[:, None].clamp_min(threshold)
    target_unit = target / target_norm[:, :, None].clamp_min(threshold)

    neighbours_np = _neighbour_tokens(token_xy, int(local_radius_tokens))
    neighbours = torch.as_tensor(neighbours_np, device=query.device, dtype=torch.long)
    present = neighbours >= 0
    safe = neighbours.clamp_min(0)
    local_target = target_unit[safe]  # [query-token, neighbour, slot, dim]
    cosine = torch.einsum("td,tnld->tnl", query_unit, local_target).clamp(-1.0, 1.0)
    local_mass = mass[safe] * valid[safe]
    # An unrelated zero-cosine surface must not receive the same positive
    # support as a meaningful appearance match.  The calibrated hinge remains
    # in [0,2], is monotone in both similarity and visible mass, and exactly
    # reduces to ``cosine+1`` when the floor is -1.
    appearance_evidence = 2.0 * torch.clamp(
        (cosine - cosine_floor) / (1.0 - cosine_floor), min=0.0, max=1.0
    )
    evidence = torch.sum(local_mass * appearance_evidence, dim=2)
    evidence = torch.where(present, evidence, torch.zeros_like(evidence))
    best_evidence, best_local = torch.max(evidence, dim=1)
    selected = torch.gather(safe, 1, best_local[:, None])[:, 0]
    token_score = torch.where(query_valid, best_evidence - 1.0, torch.full_like(best_evidence, -1.0))
    denominator = torch.sum(reliability)
    score = torch.where(
        denominator > 1.0e-12,
        torch.sum(reliability * token_score) / denominator.clamp_min(1.0e-12),
        torch.full((), -1.0, device=query.device, dtype=query.dtype),
    )
    return FullTokenSurfacePoseEnergy(
        score=score,
        token_score=token_score,
        selected_target_token=selected,
        local_radius_tokens=int(local_radius_tokens),
        minimum_cosine_evidence=cosine_floor,
        semantics=CONTRASTIVE_SEMANTICS if cosine_floor > -1.0 else SEMANTICS,
    )
