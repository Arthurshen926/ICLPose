"""Device-resident exact child/parent reduction for the soft-surface renderer.

The resident rasterizer already resolves full-scene occlusion on the GPU.  This
module keeps the resulting token hit stream on the same device and performs the
deterministic child, canonical-payload, and parent Top-L reductions there.  It
does not alter geometry, alpha compositing, hierarchy membership, or feature
codes.

The implementation deliberately uses stable lexicographic sorts followed by
segment reductions.  That mirrors the NumPy authority's canonical accumulation
orders and, unlike scatter-add based prototypes, is stable under pose-batch
permutation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class DeviceSoftChildTokenReduction:
    """Flattened token outputs; the caller owns host transfer and reshaping."""

    child_rows: object
    child_weights: object
    child_features: object
    child_feature_valid: object
    parent_rows: object
    parent_weights: object
    parent_tail_weight: object
    child_tail_weight: object
    unassigned_geometry_weight: object
    background_weight: object
    canonical_field_missing_weight: object
    payload_excluded_weight: object
    null_weight: object
    total_alpha: object
    alpha_overflow: object
    maximum_alpha_overflow: float
    overflow_token_fraction: float


def _require_aligned_device_tensors(values: tuple[object, ...]) -> None:
    import torch

    if any(not isinstance(value, torch.Tensor) for value in values):
        raise TypeError("exact device reducer inputs must be Torch tensors")
    if len({value.device for value in values}) != 1:
        raise ValueError("exact device reducer inputs must share one device")
    if any(not value.is_contiguous() for value in values):
        raise ValueError("exact device reducer inputs must be contiguous")


def _stable_lexicographic_order(*least_to_most_significant):
    """Torch equivalent of ``np.lexsort`` for aligned one-dimensional keys."""

    import torch

    if not least_to_most_significant:
        raise ValueError("lexicographic ordering requires at least one key")
    size = int(least_to_most_significant[0].numel())
    if any(value.ndim != 1 or int(value.numel()) != size for value in least_to_most_significant):
        raise ValueError("lexicographic keys must be aligned one-dimensional tensors")
    order = torch.arange(
        size, dtype=torch.int64, device=least_to_most_significant[0].device,
    )
    for key in least_to_most_significant:
        order = order[torch.argsort(key[order], stable=True)]
    return order


def _segment_sum_sorted(keys, values):
    """Sum rows with an already sorted int64 key in their existing order."""

    import torch

    if keys.ndim != 1 or keys.dtype != torch.int64 or values.shape[0] != keys.numel():
        raise ValueError("sorted segment arrays differ")
    if keys.numel() == 0:
        return keys.clone(), values[:0]
    if torch.any(keys[1:] < keys[:-1]):
        raise ValueError("segment keys must be sorted")
    starts = torch.empty_like(keys, dtype=torch.bool)
    starts[0] = True
    starts[1:] = keys[1:] != keys[:-1]
    start_rows = torch.nonzero(starts, as_tuple=False).reshape(-1)
    lengths = torch.diff(torch.cat((
        start_rows,
        torch.as_tensor([keys.numel()], dtype=torch.int64, device=keys.device),
    )))
    return keys[start_rows], torch.segment_reduce(values, "sum", lengths=lengths, axis=0)


def _rank_grouped_top_l(group_keys, grouped_mass, *, owner_count: int, token_count: int, top_l: int):
    """Rank grouped ``token*owner_count+owner`` rows by mass then owner row."""

    import torch

    owner_total = int(owner_count)
    token_total = int(token_count)
    retained = int(top_l)
    rows = torch.full(
        (token_total, retained), -1, dtype=torch.int64, device=group_keys.device,
    )
    mass = torch.zeros(
        (token_total, retained), dtype=torch.float32, device=group_keys.device,
    )
    if group_keys.numel() == 0:
        return rows, mass, group_keys, group_keys
    token = torch.div(group_keys, owner_total, rounding_mode="floor")
    owner = torch.remainder(group_keys, owner_total)
    # np.lexsort((owner, -mass, token)): owner is the final deterministic tie
    # breaker and token is the primary grouping key.
    rank = _stable_lexicographic_order(owner, -grouped_mass, token)
    ranked_token = token[rank]
    starts = torch.empty_like(ranked_token, dtype=torch.bool)
    starts[0] = True
    starts[1:] = ranked_token[1:] != ranked_token[:-1]
    start_rows = torch.nonzero(starts, as_tuple=False).reshape(-1)
    counts = torch.diff(torch.cat((
        start_rows,
        torch.as_tensor([rank.numel()], dtype=torch.int64, device=rank.device),
    )))
    within = torch.arange(rank.numel(), dtype=torch.int64, device=rank.device)
    within -= torch.repeat_interleave(start_rows, counts)
    keep = within < retained
    chosen = rank[keep]
    chosen_token = token[chosen]
    chosen_slot = within[keep]
    rows[chosen_token, chosen_slot] = owner[chosen]
    mass[chosen_token, chosen_slot] = grouped_mass[chosen].to(torch.float32)
    chosen_key = group_keys[chosen]
    chosen_flat = chosen_token * retained + chosen_slot
    key_order = torch.argsort(chosen_key, stable=True)
    return rows, mass, chosen_key[key_order], chosen_flat[key_order]


def _group_canonical_payload(
    *,
    token,
    primitive,
    weight,
    child,
    field_row,
    stable_primitive_id,
    canonical_codes,
    retained_child_keys,
    retained_flat_slots,
    child_count: int,
    output_slot_count: int,
    payload_child_mask,
    channel_block: int,
):
    """Return retained canonical alpha, payload alpha, and weighted features."""

    import torch

    feature_dim = int(canonical_codes.shape[1])
    canonical_alpha = torch.zeros(
        (output_slot_count,), dtype=torch.float32, device=token.device,
    )
    payload_alpha = torch.zeros_like(canonical_alpha)
    feature_sum = torch.zeros(
        (output_slot_count, feature_dim), dtype=torch.float32, device=token.device,
    )
    canonical = (field_row >= 0) & (child >= 0) & (weight > 0.0)
    if not torch.any(canonical) or retained_child_keys.numel() == 0:
        return canonical_alpha, payload_alpha, feature_sum

    selected = torch.nonzero(canonical, as_tuple=False).reshape(-1)
    key = token[selected] * int(child_count) + child[selected]
    position = torch.searchsorted(retained_child_keys, key)
    bounded = position.clamp_max(retained_child_keys.numel() - 1)
    present = retained_child_keys[bounded] == key
    selected = selected[present]
    position = bounded[present]
    if selected.numel() == 0:
        return canonical_alpha, payload_alpha, feature_sum
    slot = retained_flat_slots[position]
    order = _stable_lexicographic_order(
        weight[selected], stable_primitive_id[selected], slot,
    )
    selected = selected[order]
    slot = slot[order]
    unique_slot, alpha_sum = _segment_sum_sorted(slot, weight[selected])
    canonical_alpha[unique_slot] = alpha_sum.to(torch.float32)

    if payload_child_mask is None:
        payload_selected = selected
        payload_slot = slot
        payload_unique = unique_slot
        payload_sum = alpha_sum
    else:
        included = payload_child_mask[child[selected]]
        payload_selected = selected[included]
        payload_slot = slot[included]
        if payload_selected.numel() == 0:
            return canonical_alpha, payload_alpha, feature_sum
        payload_unique, payload_sum = _segment_sum_sorted(
            payload_slot, weight[payload_selected],
        )
    payload_alpha[payload_unique] = payload_sum.to(torch.float32)
    if payload_selected.numel():
        rows = field_row[payload_selected]
        for begin in range(0, feature_dim, int(channel_block)):
            end = min(begin + int(channel_block), feature_dim)
            weighted = (
                weight[payload_selected, None].to(torch.float64)
                * canonical_codes[rows, begin:end].to(torch.float64)
            )
            grouped_slot, grouped_feature = _segment_sum_sorted(payload_slot, weighted)
            feature_sum[grouped_slot, begin:end] = grouped_feature.to(torch.float32)
    return canonical_alpha, payload_alpha, feature_sum


def reduce_soft_child_token_hits_torch(
    *,
    token_ids,
    primitive_rows,
    contribution,
    stable_primitive_ids,
    child_owner_by_primitive,
    field_row_by_primitive,
    canonical_codes,
    parent_membership_offsets,
    parent_membership_rows,
    parent_membership_weights,
    stable_parent_ids,
    token_count: int,
    child_count: int,
    top_l: int,
    minimum_feature_alpha: float,
    alpha_conservation_tolerance: float,
    selected_child_rows=None,
    feature_channel_block: int = 16,
) -> DeviceSoftChildTokenReduction:
    """Reduce an occlusion-resolved hit stream without a device-to-host seam."""

    import torch

    tensors = (
        token_ids, primitive_rows, contribution, stable_primitive_ids,
        child_owner_by_primitive, field_row_by_primitive, canonical_codes,
        parent_membership_offsets, parent_membership_rows,
        parent_membership_weights, stable_parent_ids,
    )
    _require_aligned_device_tensors(tensors)
    token, primitive, weight = token_ids, primitive_rows, contribution
    if (
        token.ndim != 1 or primitive.ndim != 1 or weight.ndim != 1
        or token.shape != primitive.shape or token.shape != weight.shape
        or token.dtype != torch.int64 or primitive.dtype != torch.int64
        or weight.dtype != torch.float32
    ):
        raise ValueError("exact device hit arrays differ")
    primitive_count = int(stable_primitive_ids.numel())
    tokens = int(token_count)
    children = int(child_count)
    retained = int(top_l)
    if (
        tokens <= 0 or children <= 0 or retained <= 0
        or int(feature_channel_block) <= 0
        or stable_primitive_ids.shape != (primitive_count,)
        or stable_primitive_ids.dtype != torch.int64
        or child_owner_by_primitive.shape != (primitive_count,)
        or child_owner_by_primitive.dtype != torch.int64
        or field_row_by_primitive.shape != (primitive_count,)
        or field_row_by_primitive.dtype != torch.int64
        or canonical_codes.ndim != 2 or canonical_codes.dtype != torch.float32
        or parent_membership_offsets.shape != (primitive_count + 1,)
        or parent_membership_offsets.dtype != torch.int64
        or parent_membership_rows.ndim != 1
        or parent_membership_rows.dtype != torch.int64
        or parent_membership_weights.shape != parent_membership_rows.shape
        or parent_membership_weights.dtype != torch.float32
        or stable_parent_ids.ndim != 1 or stable_parent_ids.dtype != torch.int64
    ):
        raise ValueError("exact device hierarchy tensors differ")
    if primitive.numel() and (
        torch.any(token < 0) or torch.any(token >= tokens)
        or torch.any(primitive < 0) or torch.any(primitive >= primitive_count)
    ):
        raise ValueError("exact device hit identity is out of bounds")
    if (
        not torch.isfinite(weight).all() or torch.any(weight < 0.0)
        or not torch.isfinite(canonical_codes).all()
        or not torch.isfinite(parent_membership_weights).all()
        or torch.any(parent_membership_weights < 0.0)
    ):
        raise ValueError("exact device evidence must be finite and nonnegative")
    if parent_membership_offsets.numel() and (
        int(parent_membership_offsets[0].item()) != 0
        or int(parent_membership_offsets[-1].item()) != parent_membership_rows.numel()
        or torch.any(parent_membership_offsets[1:] < parent_membership_offsets[:-1])
    ):
        raise ValueError("exact device parent membership offsets differ")
    parent_count = int(stable_parent_ids.numel())
    if parent_membership_rows.numel() and (
        torch.any(parent_membership_rows < 0)
        or torch.any(parent_membership_rows >= parent_count)
    ):
        raise ValueError("exact device parent membership is out of bounds")

    payload_mask = None
    if selected_child_rows is not None:
        if not isinstance(selected_child_rows, torch.Tensor):
            raise TypeError("selected child rows must be a Torch tensor")
        selected = selected_child_rows.to(device=token.device, dtype=torch.int64).reshape(-1)
        if selected.numel() and (
            torch.any(selected < 0) or torch.any(selected >= children)
        ):
            raise ValueError("selected child payload rows are invalid")
        payload_mask = torch.zeros((children,), dtype=torch.bool, device=token.device)
        payload_mask[selected] = True

    stable = stable_primitive_ids[primitive]
    field_row = field_row_by_primitive[primitive]
    child = child_owner_by_primitive[primitive]
    positive_child = (child >= 0) & (child < children) & (weight > 0.0)

    total_alpha = torch.zeros((tokens,), dtype=torch.float32, device=token.device)
    assigned_alpha = torch.zeros((tokens,), dtype=torch.float64, device=token.device)
    if token.numel():
        global_order = _stable_lexicographic_order(weight, stable, token)
        grouped_token, grouped_total = _segment_sum_sorted(
            token[global_order], weight[global_order],
        )
        total_alpha[grouped_token] = grouped_total.to(torch.float32)
        assigned_order = global_order[positive_child[global_order]]
        if assigned_order.numel():
            grouped_token, grouped_assigned = _segment_sum_sorted(
                token[assigned_order], weight[assigned_order].to(torch.float64),
            )
            assigned_alpha[grouped_token] = grouped_assigned

    valid = torch.nonzero(positive_child, as_tuple=False).reshape(-1)
    if valid.numel():
        child_key = token[valid] * children + child[valid]
        child_order = _stable_lexicographic_order(
            weight[valid], stable[valid], child_key,
        )
        child_key, child_mass = _segment_sum_sorted(
            child_key[child_order], weight[valid][child_order],
        )
    else:
        child_key = torch.zeros((0,), dtype=torch.int64, device=token.device)
        child_mass = torch.zeros((0,), dtype=torch.float32, device=token.device)
    child_rows, child_weights, retained_child_keys, retained_flat_slots = (
        _rank_grouped_top_l(
            child_key, child_mass, owner_count=children,
            token_count=tokens, top_l=retained,
        )
    )

    slot_count = tokens * retained
    canonical_alpha, payload_alpha, feature_sum = _group_canonical_payload(
        token=token, primitive=primitive, weight=weight, child=child,
        field_row=field_row, stable_primitive_id=stable,
        canonical_codes=canonical_codes,
        retained_child_keys=retained_child_keys,
        retained_flat_slots=retained_flat_slots,
        child_count=children, output_slot_count=slot_count,
        payload_child_mask=payload_mask, channel_block=int(feature_channel_block),
    )
    canonical_alpha = canonical_alpha.reshape(tokens, retained)
    payload_alpha = payload_alpha.reshape(tokens, retained)
    raw_child_feature_sum = feature_sum.reshape(tokens, retained, -1)
    feature_valid = payload_alpha >= float(minimum_feature_alpha)
    normalized_child_features = (
        raw_child_feature_sum / payload_alpha[..., None].clamp_min(1.0e-8)
    )
    # NumPy's authority reduces the short feature axis in channel order.  An
    # explicit recurrence avoids a backend-dependent parallel norm tree whose
    # error is amplified for nearly cancelling feature mixtures.
    norm_square = torch.zeros_like(payload_alpha)
    for channel in range(int(normalized_child_features.shape[2])):
        norm_square += normalized_child_features[:, :, channel].square()
    norm = torch.sqrt(norm_square)[..., None]
    child_features = torch.where(
        feature_valid[..., None],
        normalized_child_features / norm.clamp_min(1.0e-8),
        raw_child_feature_sum,
    )

    # Direct primitive->parent memberships are retained independently from the
    # dominant child hierarchy, matching the CPU authority.
    membership_count = (
        parent_membership_offsets[primitive + 1] - parent_membership_offsets[primitive]
    )
    parent_hit = torch.nonzero(membership_count > 0, as_tuple=False).reshape(-1)
    if parent_hit.numel():
        counts = membership_count[parent_hit]
        start = torch.repeat_interleave(parent_membership_offsets[primitive[parent_hit]], counts)
        origin = torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts)
        membership_index = start + (
            torch.arange(int(torch.sum(counts).item()), dtype=torch.int64, device=token.device)
            - origin
        )
        expanded_token = torch.repeat_interleave(token[parent_hit], counts)
        expanded_parent = parent_membership_rows[membership_index]
        expanded_mass = (
            torch.repeat_interleave(weight[parent_hit], counts)
            * parent_membership_weights[membership_index]
        ).to(torch.float32)
        parent_key = expanded_token * parent_count + expanded_parent
        parent_order = _stable_lexicographic_order(
            expanded_mass, expanded_parent, expanded_token,
        )
        parent_key, parent_mass = _segment_sum_sorted(
            parent_key[parent_order], expanded_mass[parent_order],
        )
    else:
        parent_key = torch.zeros((0,), dtype=torch.int64, device=token.device)
        parent_mass = torch.zeros((0,), dtype=torch.float32, device=token.device)
    parent_row_index, parent_weights, _, _ = _rank_grouped_top_l(
        parent_key, parent_mass, owner_count=parent_count,
        token_count=tokens, top_l=retained,
    )
    safe_parent = parent_row_index.clamp_min(0)
    parent_rows = stable_parent_ids[safe_parent]
    parent_rows = torch.where(parent_row_index >= 0, parent_rows, parent_row_index)
    total_parent = torch.zeros((tokens,), dtype=torch.float64, device=token.device)
    if parent_key.numel():
        unique_parent_token = torch.div(parent_key, parent_count, rounding_mode="floor")
        grouped_token, grouped_parent_total = _segment_sum_sorted(
            unique_parent_token, parent_mass.to(torch.float64),
        )
        total_parent[grouped_token] = grouped_parent_total
    parent_tail = torch.clamp_min(
        total_parent - torch.sum(parent_weights.to(torch.float64), dim=1), 0.0,
    ).to(torch.float32)

    tolerance = float(alpha_conservation_tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("alpha_conservation_tolerance must be positive")
    raw_total = total_alpha.to(torch.float64)
    retained64 = child_weights.to(torch.float64)
    canonical64 = canonical_alpha.to(torch.float64)
    payload64 = payload_alpha.to(torch.float64)
    arrays = (raw_total, assigned_alpha, retained64, canonical64, payload64)
    if any(not torch.isfinite(value).all() for value in arrays):
        raise ValueError("soft child mass is nonfinite")
    if any(torch.any(value < -tolerance) for value in arrays):
        raise ValueError("soft child mass is negative")
    retained_sum = torch.sum(retained64, dim=1)
    overflow = torch.clamp_min(raw_total - 1.0, 0.0)
    if torch.any(overflow > tolerance):
        raise ValueError("rendered alpha overflow exceeds the compositing tolerance")
    if torch.any(assigned_alpha > raw_total + tolerance):
        raise ValueError("assigned child mass exceeds rendered total alpha")
    if torch.any(retained_sum > assigned_alpha + tolerance):
        raise ValueError("retained child mass exceeds assigned child mass")
    if torch.any(canonical64 > retained64 + tolerance):
        raise ValueError("canonical feature mass exceeds retained child mass")
    if torch.any(payload64 > canonical64 + tolerance):
        raise ValueError("payload feature mass exceeds canonical feature mass")
    total = torch.clamp(raw_total, 0.0, 1.0)
    assigned = torch.minimum(torch.clamp_min(assigned_alpha, 0.0), total)
    retained64 = torch.minimum(torch.clamp_min(retained64, 0.0), assigned[:, None])
    retained_sum = torch.minimum(torch.sum(retained64, dim=1), assigned)
    canonical64 = torch.minimum(torch.clamp_min(canonical64, 0.0), retained64)
    payload64 = torch.minimum(torch.clamp_min(payload64, 0.0), canonical64)
    child_tail = assigned - retained_sum
    unassigned = total - assigned
    background = 1.0 - total
    canonical_missing = torch.sum(retained64 - canonical64, dim=1)
    payload_excluded = torch.sum(canonical64 - payload64, dim=1)
    null = child_tail + unassigned + background
    if torch.any(torch.abs(retained_sum + null - 1.0) > tolerance):
        raise ValueError("soft child identity partition does not conserve unit mass")

    return DeviceSoftChildTokenReduction(
        child_rows=child_rows,
        child_weights=child_weights,
        child_features=child_features,
        child_feature_valid=feature_valid,
        parent_rows=parent_rows,
        parent_weights=parent_weights,
        parent_tail_weight=parent_tail,
        child_tail_weight=child_tail.to(torch.float32),
        unassigned_geometry_weight=unassigned.to(torch.float32),
        background_weight=background.to(torch.float32),
        canonical_field_missing_weight=canonical_missing.to(torch.float32),
        payload_excluded_weight=payload_excluded.to(torch.float32),
        null_weight=null.to(torch.float32),
        total_alpha=total.to(torch.float32),
        alpha_overflow=overflow.to(torch.float32),
        maximum_alpha_overflow=float(torch.max(overflow).item()) if overflow.numel() else 0.0,
        overflow_token_fraction=float(torch.mean((overflow > 0.0).to(torch.float32)).item()),
    )
