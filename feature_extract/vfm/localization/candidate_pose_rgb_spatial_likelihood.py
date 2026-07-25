"""Target-free high-resolution RGB spatial likelihood for frozen candidates.

The network consumes only a frozen query point, frozen top-L support-view
layout, real RGB patches, and image-space RADIO/ALIKE context.  It emits one
local ``offset + dustbin`` density per candidate/support edge.  A caller may
then evaluate that density at an externally proposed pose projection, but pose
coordinates never enter the encoder itself.

This separation is deliberate.  A projection outside the finite RGB support
window is assigned a fixed neutral factor; it must not become positive evidence
through the learned dustbin head.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from feature_extract.vfm.localization.candidate_pose_llr import (
    _FullContextCropEncoder,
    _alike_shift_correlations,
    _crop_subpixel_grid_tokens,
    bounded_log_likelihood_ratio,
    fixed_candidate_view_mixture_log_ratio,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial import (
    CandidatePoseRGBSpatialLayout,
)
from feature_extract.vfm.measurement_v1.candidate_rgb_identity_verifier import (
    normalized_spatial_log_probabilities_with_dustbin,
)
from feature_extract.vfm.measurement_v1.rgb_patch_measurement_branch import (
    TexturePatchEncoder,
    cost_volume_quality_features,
    local_offset_grid,
    template_search_cost_volume_logits,
)


CANDIDATE_POSE_RGB_SPATIAL_LIKELIHOOD_FORMAT = (
    "candidate_pose_rgb_spatial_likelihood_v2"
)
CANDIDATE_POSE_RGB_SPATIAL_OBSERVATION_PRETRAIN_CHECKPOINT_FORMAT = (
    "candidate_pose_rgb_spatial_observation_pretrain_checkpoint_v1"
)
CANDIDATE_POSE_RGB_SPATIAL_CONTEXT_OBSERVATION_PRETRAIN_CHECKPOINT_FORMAT = (
    "candidate_pose_rgb_spatial_context_observation_pretrain_checkpoint_v2"
)
CANDIDATE_POSE_RGB_SPATIAL_HARD_POSE_PRETRAIN_CHECKPOINT_FORMAT = (
    "candidate_pose_rgb_spatial_hard_pose_pretrain_checkpoint_v1"
)
CANDIDATE_POSE_RGB_SPATIAL_HARD_POSE_PRETRAIN_OBJECTIVE = (
    "candidate_specific_correct_vs_coherent_wrong_pose_group_margin_plus_"
    "correct_projection_density_and_distinct_track_dustbin_v1"
)
CANDIDATE_POSE_RGB_SPATIAL_CONTEXT_IDENTITY_PRETRAIN_OBJECTIVE = (
    "strict_observed_track_context_candidate_cross_entropy_plus_"
    "support_appearance_derangement_margin_l0_v1"
)
CANDIDATE_POSE_RGB_SPATIAL_CONTEXT_OBSERVATION_PRETRAIN_OBJECTIVE = (
    "randomized_candidate_slot_same_track_observation_context_cross_entropy_plus_"
    "geometry_fixed_support_image_derangement_margin_with_absolute_coordinate_"
    "position_control_l0_v2"
)
CANDIDATE_POSE_RGB_SPATIAL_HARD_POSE_PRETRAIN_NULL_MASS = 0.10

# These are deliberately full two-dimensional crops, not a single descriptor
# token.  RADIO carries absolute facade context; ALIKE supplies a finer spatial
# branch without replacing the RGB local likelihood.
CANDIDATE_POSE_RGB_SPATIAL_CONTEXT_WINDOWS = {
    "radio_final": 15,
    "radio_intermediate": 15,
    "alike": 13,
}
CANDIDATE_POSE_RGB_SPATIAL_CONTEXT_ENCODER_ARCHITECTURES = frozenset(
    {"conv_v1", "cross_attention_v2", "absolute_cross_attention_v3"}
)
# The context branch is trained from a neutral likelihood prior, but an exactly
# zero final projection blocks its encoder gradients on the first optimizer
# step.  This small zero-mean initialization keeps initial LLRs near zero
# while allowing direct identity/repeat supervision to reach RADIO and ALIKE.
CONTEXT_IDENTITY_HEAD_FINAL_WEIGHT_STD = 1e-2


def resolve_candidate_pose_rgb_spatial_context_windows(
    windows: Mapping[str, int] | None = None,
) -> dict[str, int]:
    """Validate the explicit per-source crop receptive fields.

    The original 15x15 RADIO crop is retained as the default for checkpoint
    compatibility.  New experiments must declare a different window through
    this function so its receptive field is serialized with the checkpoint
    instead of silently changing descriptor semantics.
    """

    values = (
        CANDIDATE_POSE_RGB_SPATIAL_CONTEXT_WINDOWS
        if windows is None
        else dict(windows)
    )
    if set(values) != set(CANDIDATE_POSE_RGB_SPATIAL_CONTEXT_WINDOWS):
        raise ValueError("candidate RGB spatial context windows are incomplete")
    resolved = {name: int(values[name]) for name in CANDIDATE_POSE_RGB_SPATIAL_CONTEXT_WINDOWS}
    if any(value < 3 or value % 2 == 0 for value in resolved.values()):
        raise ValueError("candidate RGB spatial context windows must be odd and at least three")
    return resolved


def resolve_candidate_pose_rgb_spatial_context_encoder_arch(value: str | None = None) -> str:
    """Resolve an explicit context encoder architecture for checkpoint lineage.

    ``conv_v1`` preserves the initial convolutional crop encoder exactly.
    ``cross_attention_v2`` adds only local crop-relative positions.
    ``absolute_cross_attention_v3`` additionally encodes each sampled token's
    normalized coordinates in its own full image; this is the explicit
    absolute-phase experiment and must never be conflated with either older
    relative-context checkpoint.
    """

    name = "conv_v1" if value is None else str(value).strip().lower()
    if name not in CANDIDATE_POSE_RGB_SPATIAL_CONTEXT_ENCODER_ARCHITECTURES:
        raise ValueError("candidate RGB spatial context encoder architecture is invalid")
    return name


@dataclass(frozen=True)
class CandidatePoseRGBSpatialRuntime:
    """Immutable target-free point/candidate/support layout for one scorer call.

    The runtime intentionally omits track identities, rank/coarse scores,
    target labels, residuals, and pose matrices.  Candidate priors are only
    used outside the encoder by the fixed mixture function.
    """

    query_image_indices: torch.Tensor
    query_xy: torch.Tensor
    support_image_indices: torch.Tensor
    support_xy: torch.Tensor
    support_view_valid: torch.Tensor
    candidate_view_weights: torch.Tensor
    candidate_probabilities: torch.Tensor
    null_probabilities: torch.Tensor

    def __post_init__(self) -> None:
        query_indices = torch.as_tensor(self.query_image_indices, dtype=torch.long).reshape(-1)
        query_xy = torch.as_tensor(self.query_xy, dtype=torch.float32)
        support_indices = torch.as_tensor(self.support_image_indices, dtype=torch.long)
        support_xy = torch.as_tensor(self.support_xy, dtype=torch.float32)
        support_valid = torch.as_tensor(self.support_view_valid, dtype=torch.bool)
        view_weights = torch.as_tensor(self.candidate_view_weights, dtype=torch.float32)
        candidates = torch.as_tensor(self.candidate_probabilities, dtype=torch.float32)
        null = torch.as_tensor(self.null_probabilities, dtype=torch.float32).reshape(-1)
        if (
            len(query_indices) == 0
            or query_xy.shape != (len(query_indices), 2)
            or support_indices.ndim != 3
            or support_indices.shape[0] != len(query_indices)
            or support_xy.shape != (*support_indices.shape, 2)
            or support_valid.shape != support_indices.shape
            or view_weights.shape != support_indices.shape
            or candidates.shape != support_indices.shape[:2]
            or null.shape != (len(query_indices),)
            or torch.any(query_indices < 0)
            or torch.any(support_indices < 0)
            or not torch.isfinite(query_xy).all()
            or not torch.isfinite(support_xy).all()
            or not torch.isfinite(view_weights).all()
            or not torch.isfinite(candidates).all()
            or not torch.isfinite(null).all()
            or torch.any(view_weights < 0.0)
            or torch.any(candidates < 0.0)
            or torch.any(null < 0.0)
        ):
            raise ValueError("candidate RGB spatial runtime arrays are invalid")
        if torch.any(torch.abs(candidates.sum(dim=1) + null - 1.0) > 1e-4):
            raise ValueError("candidate RGB spatial runtime priors must sum to one")
        positive = candidates > 0.0
        view_mass = view_weights.sum(dim=2)
        if (
            torch.any(torch.abs(view_mass[positive] - 1.0) > 1e-4)
            or torch.any(view_mass[~positive] > 1e-6)
            or torch.any(positive & ~torch.any(support_valid, dim=2))
        ):
            raise ValueError("candidate RGB spatial runtime view weights are invalid")
        object.__setattr__(self, "query_image_indices", query_indices)
        object.__setattr__(self, "query_xy", query_xy)
        object.__setattr__(self, "support_image_indices", support_indices)
        object.__setattr__(self, "support_xy", support_xy)
        object.__setattr__(self, "support_view_valid", support_valid)
        object.__setattr__(self, "candidate_view_weights", view_weights)
        object.__setattr__(self, "candidate_probabilities", candidates)
        object.__setattr__(self, "null_probabilities", null)

    @property
    def point_count(self) -> int:
        return int(self.query_image_indices.numel())

    @property
    def candidate_count(self) -> int:
        return int(self.support_image_indices.shape[1])

    @property
    def support_view_count(self) -> int:
        return int(self.support_image_indices.shape[2])

    def to(self, device: torch.device | str) -> "CandidatePoseRGBSpatialRuntime":
        """Move fixed target-free tensors without attaching supervision."""

        return CandidatePoseRGBSpatialRuntime(
            query_image_indices=self.query_image_indices.to(device=device),
            query_xy=self.query_xy.to(device=device),
            support_image_indices=self.support_image_indices.to(device=device),
            support_xy=self.support_xy.to(device=device),
            support_view_valid=self.support_view_valid.to(device=device),
            candidate_view_weights=self.candidate_view_weights.to(device=device),
            candidate_probabilities=self.candidate_probabilities.to(device=device),
            null_probabilities=self.null_probabilities.to(device=device),
        )


def runtime_from_target_free_layout(
    layout: CandidatePoseRGBSpatialLayout,
    *,
    image_ids: Sequence[str] | np.ndarray,
) -> CandidatePoseRGBSpatialRuntime:
    """Join a validated target-free layout to a frozen image-ID table.

    This deliberately accepts only :class:`CandidatePoseRGBSpatialLayout`.
    Passing a train-only target artifact cannot silently leak supervision into a
    runtime scorer merely because it has similarly shaped arrays.
    """

    if not isinstance(layout, CandidatePoseRGBSpatialLayout):
        raise ValueError("runtime construction requires a target-free layout")
    ids = np.asarray(image_ids).astype(str).reshape(-1)
    if len(ids) == 0 or np.any(ids == "") or len(set(ids.tolist())) != len(ids):
        raise ValueError("runtime image IDs are invalid")
    index_by_id = {image_id: index for index, image_id in enumerate(ids.tolist())}
    try:
        query_indices = np.asarray(
            [index_by_id[str(image_id)] for image_id in layout.query_ids.tolist()],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError("runtime query image is absent from context sources") from error
    support_indices = np.zeros(layout.support_image_ids.shape, dtype=np.int64)
    for flat_index, image_id in enumerate(layout.support_image_ids.reshape(-1).tolist()):
        if not image_id:
            continue
        try:
            support_indices.reshape(-1)[flat_index] = index_by_id[str(image_id)]
        except KeyError as error:
            raise ValueError("runtime support image is absent from context sources") from error
    return CandidatePoseRGBSpatialRuntime(
        query_image_indices=torch.from_numpy(query_indices),
        query_xy=torch.from_numpy(np.asarray(layout.xy, dtype=np.float32)),
        support_image_indices=torch.from_numpy(support_indices),
        support_xy=torch.from_numpy(np.asarray(layout.support_xy, dtype=np.float32)),
        support_view_valid=torch.from_numpy(np.asarray(layout.support_view_valid, dtype=bool)),
        candidate_view_weights=torch.from_numpy(
            np.asarray(layout.support_view_weights, dtype=np.float32)
        ),
        candidate_probabilities=torch.from_numpy(
            np.asarray(layout.candidate_prior_probabilities, dtype=np.float32)
        ),
        null_probabilities=torch.from_numpy(
            np.asarray(layout.null_probabilities, dtype=np.float32)
        ),
    )


def permute_runtime_support_appearance(
    runtime: CandidatePoseRGBSpatialRuntime,
    *,
    shift: int = 1,
) -> CandidatePoseRGBSpatialRuntime:
    """Cyclically permute valid support-image/coordinate appearances per query.

    This is a train-only visual permutation control.  It preserves the query
    anchors, candidate prior/null mass, support validity, and mixture weights;
    only the support appearance pairing is shuffled.  Layout guarantees that
    no valid support is the query image, so a within-query permutation retains
    that exclusion.
    """

    if not isinstance(runtime, CandidatePoseRGBSpatialRuntime):
        raise ValueError("support permutation requires a target-free runtime")
    indices = runtime.support_image_indices.clone()
    coordinates = runtime.support_xy.clone()
    valid = runtime.support_view_valid
    for point_index in range(runtime.point_count):
        slots = torch.nonzero(valid[point_index], as_tuple=False)
        if len(slots) <= 1:
            continue
        amount = int(shift) % int(len(slots))
        if amount == 0:
            continue
        rows = slots[:, 0]
        columns = slots[:, 1]
        indices[point_index, rows, columns] = torch.roll(
            indices[point_index, rows, columns], shifts=amount, dims=0
        )
        coordinates[point_index, rows, columns] = torch.roll(
            coordinates[point_index, rows, columns], shifts=amount, dims=0
        )
    return CandidatePoseRGBSpatialRuntime(
        query_image_indices=runtime.query_image_indices,
        query_xy=runtime.query_xy,
        support_image_indices=indices,
        support_xy=coordinates,
        support_view_valid=runtime.support_view_valid,
        candidate_view_weights=runtime.candidate_view_weights,
        candidate_probabilities=runtime.candidate_probabilities,
        null_probabilities=runtime.null_probabilities,
    )


def permute_runtime_support_image_appearance_only(
    runtime: CandidatePoseRGBSpatialRuntime,
    *,
    shift: int = 1,
) -> CandidatePoseRGBSpatialRuntime:
    """Derange support image content while preserving all geometry fields.

    Descriptor-grid context branches crop a fixed support coordinate from an
    image-indexed feature map.  Moving both the image and coordinate therefore
    changes not only appearance but also border validity and local geometry.
    This stricter control rolls only valid support *image indices*; query
    anchors, support coordinates, validity, candidate/null mass, and immutable
    view weights stay byte-identical.  Callers must use a common image
    coordinate frame and verify their visual availability masks remain equal.
    """

    if not isinstance(runtime, CandidatePoseRGBSpatialRuntime):
        raise ValueError("support image appearance permutation requires a target-free runtime")
    indices = runtime.support_image_indices.clone()
    valid = runtime.support_view_valid
    for point_index in range(runtime.point_count):
        slots = torch.nonzero(valid[point_index], as_tuple=False)
        if len(slots) <= 1:
            continue
        amount = int(shift) % int(len(slots))
        if amount == 0:
            continue
        rows = slots[:, 0]
        columns = slots[:, 1]
        indices[point_index, rows, columns] = torch.roll(
            indices[point_index, rows, columns], shifts=amount, dims=0
        )
    return CandidatePoseRGBSpatialRuntime(
        query_image_indices=runtime.query_image_indices,
        query_xy=runtime.query_xy,
        support_image_indices=indices,
        support_xy=runtime.support_xy,
        support_view_valid=runtime.support_view_valid,
        candidate_view_weights=runtime.candidate_view_weights,
        candidate_probabilities=runtime.candidate_probabilities,
        null_probabilities=runtime.null_probabilities,
    )


def permute_support_patch_appearance(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    support_patches: torch.Tensor,
    shift: int = 1,
) -> torch.Tensor:
    """Apply the runtime's support-only derangement to already cropped RGB.

    ``permute_runtime_support_appearance`` moves the complete image/coordinate
    appearance tuple over valid candidate-view slots while leaving candidate
    priors and geometry fixed.  RGB crops are a deterministic function of that
    tuple, so reindexing the already cropped tensor is exactly equivalent to a
    second expensive crop pass.  Invalid slots remain zeroed and never become
    a fallback visual signal.
    """

    if not isinstance(runtime, CandidatePoseRGBSpatialRuntime):
        raise ValueError("support patch appearance permutation requires a target-free runtime")
    patches = torch.as_tensor(support_patches)
    expected_prefix = tuple(int(value) for value in runtime.support_image_indices.shape)
    if patches.ndim < 4 or tuple(int(value) for value in patches.shape[:3]) != expected_prefix:
        raise ValueError("support patch tensor does not align with runtime support slots")
    valid = runtime.support_view_valid.to(device=patches.device, dtype=torch.bool)
    out = torch.zeros_like(patches)
    for point_index in range(runtime.point_count):
        slots = torch.nonzero(valid[point_index], as_tuple=False)
        if len(slots) == 0:
            continue
        rows = slots[:, 0]
        columns = slots[:, 1]
        amount = int(shift) % int(len(slots))
        out[point_index, rows, columns] = torch.roll(
            patches[point_index, rows, columns], shifts=amount, dims=0
        )
    return out


def permute_runtime_candidate_slots(
    runtime: CandidatePoseRGBSpatialRuntime, *, permutations: torch.Tensor
) -> CandidatePoseRGBSpatialRuntime:
    """Reorder complete candidate slots without exposing a slot identity.

    This is useful only while fitting train-only candidate labels.  Every
    candidate-owned target-free field moves together, while the explicit null
    mass and query appearance remain unchanged.  A caller must transform its
    label after this function returns; labels are deliberately not accepted
    here.
    """

    if not isinstance(runtime, CandidatePoseRGBSpatialRuntime):
        raise ValueError("candidate slot permutation requires a target-free runtime")
    order = torch.as_tensor(permutations, dtype=torch.long, device=runtime.query_xy.device)
    point_count = runtime.point_count
    candidate_count = runtime.candidate_count
    if (
        order.shape != (point_count, candidate_count)
        or torch.any(order < 0)
        or torch.any(order >= candidate_count)
        or not torch.equal(
            torch.sort(order, dim=1).values,
            torch.arange(candidate_count, device=order.device).expand(point_count, -1),
        )
    ):
        raise ValueError("candidate slot permutations are invalid")
    view_count = runtime.support_view_count
    slot_view = order.unsqueeze(-1).expand(-1, -1, view_count)
    slot_xy = slot_view.unsqueeze(-1).expand(-1, -1, -1, 2)
    return CandidatePoseRGBSpatialRuntime(
        query_image_indices=runtime.query_image_indices,
        query_xy=runtime.query_xy,
        support_image_indices=runtime.support_image_indices.gather(1, slot_view),
        support_xy=runtime.support_xy.gather(1, slot_xy),
        support_view_valid=runtime.support_view_valid.gather(1, slot_view),
        candidate_view_weights=runtime.candidate_view_weights.gather(1, slot_view),
        candidate_probabilities=runtime.candidate_probabilities.gather(1, order),
        null_probabilities=runtime.null_probabilities,
    )


def _validate_offset_grid(offsets_xy: torch.Tensor) -> tuple[int, float, float, float, float]:
    offsets = torch.as_tensor(offsets_xy, dtype=torch.float32)
    if offsets.ndim != 2 or offsets.shape[1] != 2 or len(offsets) == 0 or not torch.isfinite(offsets).all():
        raise ValueError("local spatial offset grid is invalid")
    side = int(round(math.sqrt(int(len(offsets)))))
    if side * side != int(len(offsets)) or side < 2:
        raise ValueError("local spatial offset grid must be a square with at least two cells")
    xs = torch.unique(offsets[:, 0], sorted=True)
    ys = torch.unique(offsets[:, 1], sorted=True)
    if len(xs) != side or len(ys) != side:
        raise ValueError("local spatial offset grid is not Cartesian")
    step_x = float((xs[1] - xs[0]).item())
    step_y = float((ys[1] - ys[0]).item())
    if step_x <= 0.0 or step_y <= 0.0 or not math.isclose(step_x, step_y, rel_tol=1e-5, abs_tol=1e-5):
        raise ValueError("local spatial offset grid step is invalid")
    expected_x, expected_y = torch.meshgrid(xs, ys, indexing="xy")
    expected = torch.stack([expected_x.reshape(-1), expected_y.reshape(-1)], dim=1)
    if not torch.allclose(offsets.cpu(), expected.cpu(), atol=1e-5, rtol=1e-5):
        raise ValueError("local spatial offset grid ordering is invalid")
    return side, float(xs[0].item()), float(ys[0].item()), step_x, float(xs[-1].item())


@dataclass(frozen=True)
class CandidatePoseRGBSpatialEdgePrediction:
    """Target-free density and context outputs for every fixed support edge."""

    spatial_logits: torch.Tensor
    non_dustbin_logits: torch.Tensor
    joint_log_probabilities: torch.Tensor
    offsets_xy: torch.Tensor
    context_log_likelihood_ratios: torch.Tensor
    edge_usable: torch.Tensor
    raw_spatial_logits: torch.Tensor | None = None
    spatial_residual_logits: torch.Tensor | None = None
    rgb_edge_usable: torch.Tensor | None = None
    context_edge_usable: torch.Tensor | None = None

    def __post_init__(self) -> None:
        spatial = torch.as_tensor(self.spatial_logits, dtype=torch.float32)
        non_dustbin = torch.as_tensor(self.non_dustbin_logits, dtype=torch.float32)
        joint = torch.as_tensor(self.joint_log_probabilities, dtype=torch.float32)
        offsets = torch.as_tensor(self.offsets_xy, dtype=torch.float32)
        context = torch.as_tensor(self.context_log_likelihood_ratios, dtype=torch.float32)
        usable = torch.as_tensor(self.edge_usable, dtype=torch.bool)
        rgb_usable = (
            None
            if self.rgb_edge_usable is None
            else torch.as_tensor(self.rgb_edge_usable, dtype=torch.bool)
        )
        context_usable = (
            None
            if self.context_edge_usable is None
            else torch.as_tensor(self.context_edge_usable, dtype=torch.bool)
        )
        raw_spatial = (
            None
            if self.raw_spatial_logits is None
            else torch.as_tensor(self.raw_spatial_logits, dtype=torch.float32)
        )
        spatial_residual = (
            None
            if self.spatial_residual_logits is None
            else torch.as_tensor(self.spatial_residual_logits, dtype=torch.float32)
        )
        if (
            spatial.ndim != 4
            or spatial.shape[-1] < 4
            or non_dustbin.shape != spatial.shape[:-1]
            or joint.shape != (*spatial.shape[:-1], spatial.shape[-1] + 1)
            or offsets.shape != (spatial.shape[-1], 2)
            or context.shape != spatial.shape[:-1]
            or usable.shape != spatial.shape[:-1]
            or not torch.isfinite(spatial).all()
            or not torch.isfinite(non_dustbin).all()
            or not torch.isfinite(joint).all()
            or not torch.isfinite(context).all()
            or (raw_spatial is None) != (spatial_residual is None)
            or (raw_spatial is not None and raw_spatial.shape != spatial.shape)
            or (spatial_residual is not None and spatial_residual.shape != spatial.shape)
            or (raw_spatial is not None and not torch.isfinite(raw_spatial).all())
            or (spatial_residual is not None and not torch.isfinite(spatial_residual).all())
        ):
            raise ValueError("candidate RGB spatial edge prediction is invalid")
        if rgb_usable is None and context_usable is None:
            # Hand-authored diagnostic predictions from the original format
            # have one shared mask. Preserve that meaning explicitly while
            # production forwards use the source-specific fields below.
            rgb_usable = usable
            context_usable = usable
        else:
            if rgb_usable is None:
                rgb_usable = torch.zeros_like(usable)
            if context_usable is None:
                context_usable = torch.zeros_like(usable)
            if (
                rgb_usable.shape != usable.shape
                or context_usable.shape != usable.shape
                or not torch.equal(usable, rgb_usable | context_usable)
            ):
                raise ValueError(
                    "candidate RGB spatial source masks must union to the combined mask"
                )
        _validate_offset_grid(offsets)
        if torch.any(torch.abs(torch.exp(joint).sum(dim=-1) - 1.0) > 1e-4):
            raise ValueError("candidate RGB spatial edge density is not normalized")
        object.__setattr__(self, "spatial_logits", spatial)
        object.__setattr__(self, "non_dustbin_logits", non_dustbin)
        object.__setattr__(self, "joint_log_probabilities", joint)
        object.__setattr__(self, "offsets_xy", offsets)
        object.__setattr__(self, "context_log_likelihood_ratios", context)
        object.__setattr__(self, "edge_usable", usable)
        object.__setattr__(self, "raw_spatial_logits", raw_spatial)
        object.__setattr__(self, "spatial_residual_logits", spatial_residual)
        object.__setattr__(self, "rgb_edge_usable", rgb_usable)
        object.__setattr__(self, "context_edge_usable", context_usable)


def candidate_pose_rgb_spatial_component_edge_usable(
    *,
    prediction: CandidatePoseRGBSpatialEdgePrediction,
    component: str,
) -> torch.Tensor:
    """Return the source-valid mask for one target-free score component.

    The raw RGB cost volume is meaningful whenever both real RGB patches are
    in-bounds.  The learned residual/dustbin heads consume both RGB quality and
    RADIO/ALIKE context, so they require their intersection.  Context-only
    candidate identity has its own independent support.  Keeping these masks
    separate prevents a missing source from silently deleting valid evidence
    from the other source.
    """

    if not isinstance(prediction, CandidatePoseRGBSpatialEdgePrediction):
        raise ValueError("candidate RGB spatial component mask requires an edge prediction")
    name = str(component).strip().lower()
    rgb = prediction.rgb_edge_usable
    context = prediction.context_edge_usable
    assert rgb is not None and context is not None
    if name == "combined":
        return prediction.edge_usable
    if name == "rgb_cost_volume":
        return rgb
    if name in {
        "rgb_cost_volume_with_dustbin",
        "learned_spatial_no_context",
        "spatial_residual",
    }:
        return rgb & context
    if name == "context_only":
        return context
    raise ValueError(f"unknown candidate RGB spatial score component: {component!r}")


def _source_safe_joint_log_probabilities(
    *,
    full_joint_log_probabilities: torch.Tensor,
    raw_spatial_logits: torch.Tensor,
    rgb_edge_usable: torch.Tensor,
    context_edge_usable: torch.Tensor,
) -> torch.Tensor:
    """Choose a normalized density without crossing unavailable sources.

    The learned residual/dustbin density is a joint RGB-plus-context head.  If
    only RGB is valid, use the raw cost volume with a fixed-neutral dustbin; if
    only context is valid, use a fully neutral local density so that the scalar
    context LLR remains the sole evidence.  This leaves every fallback
    normalized and prevents padded RGB or synthetic context crops from entering
    a pose score.
    """

    full = torch.as_tensor(full_joint_log_probabilities, dtype=torch.float32)
    raw = torch.as_tensor(raw_spatial_logits, dtype=torch.float32, device=full.device)
    rgb = torch.as_tensor(rgb_edge_usable, dtype=torch.bool, device=full.device)
    context = torch.as_tensor(context_edge_usable, dtype=torch.bool, device=full.device)
    if (
        full.ndim != 4
        or raw.shape != full.shape[:-1] + (full.shape[-1] - 1,)
        or rgb.shape != full.shape[:-1]
        or context.shape != full.shape[:-1]
    ):
        raise ValueError("source-safe RGB spatial density inputs are incompatible")
    raw_joint = normalized_spatial_log_probabilities_with_dustbin(
        raw.reshape(-1, raw.shape[-1]),
        torch.zeros(raw.shape[:-1].numel(), dtype=raw.dtype, device=raw.device),
    ).reshape_as(full)
    neutral_spatial = torch.zeros_like(raw)
    neutral_joint = normalized_spatial_log_probabilities_with_dustbin(
        neutral_spatial.reshape(-1, neutral_spatial.shape[-1]),
        torch.zeros(
            neutral_spatial.shape[:-1].numel(),
            dtype=neutral_spatial.dtype,
            device=neutral_spatial.device,
        ),
    ).reshape_as(full)
    fusion_usable = rgb & context
    return torch.where(
        fusion_usable.unsqueeze(-1),
        full,
        torch.where(rgb.unsqueeze(-1), raw_joint, neutral_joint),
    )


def candidate_pose_rgb_spatial_score_component_prediction(
    *,
    prediction: CandidatePoseRGBSpatialEdgePrediction,
    component: str,
) -> CandidatePoseRGBSpatialEdgePrediction:
    """Return an auditable target-free score component with neutralized peers.

    The function never accepts a pose projection or target.  It only rebuilds
    normalized local densities from a model's already-emitted target-free
    tensors, so normal-versus-support-permuted component audits retain exactly
    the same geometry and candidate mixture as the combined scorer.
    """

    if not isinstance(prediction, CandidatePoseRGBSpatialEdgePrediction):
        raise ValueError("candidate RGB spatial component audit requires an edge prediction")
    name = str(component).strip().lower()
    if name == "combined":
        return prediction
    zero_spatial = torch.zeros_like(prediction.spatial_logits)
    zero_non_dustbin = torch.zeros_like(prediction.non_dustbin_logits)
    zero_context = torch.zeros_like(prediction.context_log_likelihood_ratios)
    raw = prediction.raw_spatial_logits
    residual = prediction.spatial_residual_logits
    component_usable = candidate_pose_rgb_spatial_component_edge_usable(
        prediction=prediction,
        component=name,
    )
    if name in {"rgb_cost_volume", "rgb_cost_volume_with_dustbin", "spatial_residual"} and (
        raw is None or residual is None
    ):
        raise ValueError("candidate RGB spatial prediction lacks decomposed cost-volume logits")
    if name == "rgb_cost_volume":
        spatial_logits = raw
        non_dustbin_logits = zero_non_dustbin
        context = zero_context
    elif name == "rgb_cost_volume_with_dustbin":
        spatial_logits = raw
        non_dustbin_logits = prediction.non_dustbin_logits
        context = zero_context
    elif name == "learned_spatial_no_context":
        spatial_logits = prediction.spatial_logits
        non_dustbin_logits = prediction.non_dustbin_logits
        context = zero_context
    elif name == "spatial_residual":
        spatial_logits = residual
        non_dustbin_logits = zero_non_dustbin
        context = zero_context
    elif name == "context_only":
        spatial_logits = zero_spatial
        non_dustbin_logits = zero_non_dustbin
        context = prediction.context_log_likelihood_ratios
    else:
        raise ValueError(f"unknown candidate RGB spatial score component: {component!r}")
    assert spatial_logits is not None
    joint = normalized_spatial_log_probabilities_with_dustbin(
        spatial_logits.reshape(-1, spatial_logits.shape[-1]),
        non_dustbin_logits.reshape(-1),
    ).reshape_as(prediction.joint_log_probabilities)
    rgb_usable = torch.zeros_like(component_usable)
    context_usable = torch.zeros_like(component_usable)
    if name == "context_only":
        context_usable = component_usable
    else:
        # The component representation is scored as one self-contained
        # appearance source. In particular, the learned dustbin/residual path
        # is only exposed where both original sources were present.
        rgb_usable = component_usable
    return CandidatePoseRGBSpatialEdgePrediction(
        spatial_logits=spatial_logits,
        non_dustbin_logits=non_dustbin_logits,
        joint_log_probabilities=joint,
        offsets_xy=prediction.offsets_xy,
        context_log_likelihood_ratios=context,
        edge_usable=component_usable,
        raw_spatial_logits=raw,
        spatial_residual_logits=residual,
        rgb_edge_usable=rgb_usable,
        context_edge_usable=context_usable,
    )


def continuous_joint_log_probability_at_offsets(
    *,
    joint_log_probabilities: torch.Tensor,
    offsets_xy: torch.Tensor,
    query_offsets_xy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinearly evaluate a local joint density at candidate pose offsets.

    The returned mask says that an offset lies in the finite local grid.  The
    caller must use that mask rather than interpreting the learned dustbin as
    evidence for an out-of-window pose.
    """

    joint = torch.as_tensor(joint_log_probabilities, dtype=torch.float32)
    offsets = torch.as_tensor(offsets_xy, dtype=torch.float32, device=joint.device)
    query = torch.as_tensor(query_offsets_xy, dtype=torch.float32, device=joint.device)
    if joint.ndim != 4 or joint.shape[-1] < 5 or query.ndim != 4 or query.shape[-1] != 2:
        raise ValueError("continuous spatial likelihood inputs are invalid")
    point_count, candidate_count, view_count, category_count = joint.shape
    if query.shape[1:3] != (point_count, candidate_count) or query.shape[0] == 0:
        raise ValueError("candidate projection offsets do not match edge density")
    if not torch.isfinite(query).all():
        raise ValueError("candidate projection offsets are not finite")
    side, minimum_x, minimum_y, step, maximum_x = _validate_offset_grid(offsets)
    maximum_y = float(offsets[:, 1].max().item())
    values = torch.exp(joint[..., :-1]).reshape(point_count, candidate_count, view_count, side, side)
    targets = query[:, :, :, None, :].expand(-1, -1, -1, view_count, -1)
    epsilon = max(1e-5, step * 1e-5)
    in_window = (
        (targets[..., 0] >= minimum_x - epsilon)
        & (targets[..., 0] <= maximum_x + epsilon)
        & (targets[..., 1] >= minimum_y - epsilon)
        & (targets[..., 1] <= maximum_y + epsilon)
    )
    column = ((targets[..., 0] - minimum_x) / step).clamp(0.0, float(side - 1))
    row = ((targets[..., 1] - minimum_y) / step).clamp(0.0, float(side - 1))
    column0 = torch.floor(column).to(dtype=torch.long)
    row0 = torch.floor(row).to(dtype=torch.long)
    column1 = (column0 + 1).clamp_max(side - 1)
    row1 = (row0 + 1).clamp_max(side - 1)
    fraction_x = torch.where(column0 == column1, torch.zeros_like(column), column - column0)
    fraction_y = torch.where(row0 == row1, torch.zeros_like(row), row - row0)

    batch_size = int(query.shape[0])
    flat = values.reshape(1, point_count, candidate_count, view_count, side * side).expand(
        batch_size, -1, -1, -1, -1
    )

    def gather(rows: torch.Tensor, columns: torch.Tensor) -> torch.Tensor:
        linear = (rows * side + columns).unsqueeze(-1)
        return torch.gather(flat, dim=4, index=linear).squeeze(-1)

    value00 = gather(row0, column0)
    value10 = gather(row0, column1)
    value01 = gather(row1, column0)
    value11 = gather(row1, column1)
    probability = (
        (1.0 - fraction_x) * (1.0 - fraction_y) * value00
        + fraction_x * (1.0 - fraction_y) * value10
        + (1.0 - fraction_x) * fraction_y * value01
        + fraction_x * fraction_y * value11
    )
    return torch.log(probability.clamp_min(torch.finfo(probability.dtype).tiny)), in_window


def edge_log_likelihood_ratio_at_pose_projection(
    *,
    prediction: CandidatePoseRGBSpatialEdgePrediction,
    candidate_projection_offsets_xy: torch.Tensor,
    candidate_projection_valid: torch.Tensor,
    missing_edge_log_likelihood_ratio: float = 0.0,
    max_abs_log_likelihood_ratio: float = 6.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate target-free edge evidence for caller-provided projections.

    Invalid or out-of-window projections receive exactly the fixed missing
    value.  In particular, the learned dustbin probability is never consulted
    in that case.
    """

    missing = float(missing_edge_log_likelihood_ratio)
    cap = float(max_abs_log_likelihood_ratio)
    if not math.isfinite(missing) or not math.isfinite(cap) or cap <= 0.0:
        raise ValueError("edge likelihood ratio constants are invalid")
    projected = torch.as_tensor(
        candidate_projection_offsets_xy,
        dtype=torch.float32,
        device=prediction.joint_log_probabilities.device,
    )
    projection_valid = torch.as_tensor(
        candidate_projection_valid,
        dtype=torch.bool,
        device=prediction.joint_log_probabilities.device,
    )
    if (
        projected.ndim != 4
        or projected.shape[-1] != 2
        or projection_valid.shape != projected.shape[:-1]
        or projected.shape[1:3]
        != prediction.joint_log_probabilities.shape[:2]
    ):
        raise ValueError("candidate pose projection inputs are invalid")
    local_log_probability, in_window = continuous_joint_log_probability_at_offsets(
        joint_log_probabilities=prediction.joint_log_probabilities,
        offsets_xy=prediction.offsets_xy,
        query_offsets_xy=projected,
    )
    category_count = int(prediction.joint_log_probabilities.shape[-1] - 1)
    neutral_local_log_probability = -math.log(2.0 * float(category_count))
    raw = (
        local_log_probability
        - neutral_local_log_probability
        + prediction.context_log_likelihood_ratios.unsqueeze(0)
    )
    bounded = bounded_log_likelihood_ratio(raw, max_abs_log_ratio=cap)
    usable = (
        projection_valid.unsqueeze(-1)
        & in_window
        & prediction.edge_usable.unsqueeze(0)
    )
    return torch.where(usable, bounded, torch.full_like(bounded, missing)), usable


def selected_candidate_view_log_likelihood_ratio_at_offsets(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidatePoseRGBSpatialEdgePrediction,
    point_indices: torch.Tensor,
    candidate_indices: torch.Tensor,
    offsets_xy: torch.Tensor,
    projection_valid: torch.Tensor | None = None,
    missing_edge_log_likelihood_ratio: float = 0.0,
    max_abs_log_likelihood_ratio: float = 6.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Marginalize selected target-free candidate views at supplied offsets.

    This is intentionally a post-encoder operation.  The model never receives
    point/candidate labels or offsets as inputs, while train-only code may use
    it to compare a correct candidate edge with a coherent-wrong edge.
    """

    if not isinstance(runtime, CandidatePoseRGBSpatialRuntime):
        raise ValueError("selected candidate scoring requires a target-free runtime")
    missing = float(missing_edge_log_likelihood_ratio)
    cap = float(max_abs_log_likelihood_ratio)
    if not math.isfinite(missing) or not math.isfinite(cap) or cap <= 0.0:
        raise ValueError("selected candidate scoring constants are invalid")
    device = prediction.joint_log_probabilities.device
    active = runtime.to(device)
    points = torch.as_tensor(point_indices, dtype=torch.long, device=device).reshape(-1)
    candidates = torch.as_tensor(candidate_indices, dtype=torch.long, device=device).reshape(-1)
    offsets = torch.as_tensor(offsets_xy, dtype=torch.float32, device=device)
    if projection_valid is None:
        valid = torch.ones((len(points),), dtype=torch.bool, device=device)
    else:
        valid = torch.as_tensor(projection_valid, dtype=torch.bool, device=device).reshape(-1)
    if (
        len(points) == 0
        or candidates.shape != points.shape
        or offsets.shape != (len(points), 2)
        or valid.shape != points.shape
        or torch.any(points < 0)
        or torch.any(points >= active.point_count)
        or torch.any(candidates < 0)
        or torch.any(candidates >= active.candidate_count)
        or not torch.isfinite(offsets).all()
        or prediction.joint_log_probabilities.shape[:3]
        != active.support_image_indices.shape
    ):
        raise ValueError("selected candidate scoring inputs are invalid")
    selected_joint = prediction.joint_log_probabilities[points, candidates].unsqueeze(1)
    selected_offsets = offsets.reshape(1, len(points), 1, 2)
    local_log_probability, in_window = continuous_joint_log_probability_at_offsets(
        joint_log_probabilities=selected_joint,
        offsets_xy=prediction.offsets_xy,
        query_offsets_xy=selected_offsets,
    )
    category_count = int(selected_joint.shape[-1] - 1)
    neutral_local_log_probability = -math.log(2.0 * float(category_count))
    context = prediction.context_log_likelihood_ratios[points, candidates]
    raw = local_log_probability[0, :, 0] - neutral_local_log_probability + context
    bounded = bounded_log_likelihood_ratio(raw, max_abs_log_ratio=cap)
    edge_usable = (
        valid[:, None]
        & in_window[0, :, 0]
        & prediction.edge_usable[points, candidates]
    )
    effective = torch.where(edge_usable, bounded, torch.full_like(bounded, missing))
    weights = active.candidate_view_weights[points, candidates]
    safe_log_weights = torch.where(
        weights > 0.0,
        torch.log(weights),
        torch.full_like(weights, -torch.inf),
    )
    score = torch.logsumexp(effective + safe_log_weights, dim=1)
    candidate_usable = valid & torch.any(edge_usable & (weights > 0.0), dim=1)
    return score, candidate_usable


def selected_candidate_view_context_log_likelihood_ratio(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidatePoseRGBSpatialEdgePrediction,
    point_indices: torch.Tensor,
    candidate_indices: torch.Tensor,
    missing_edge_log_likelihood_ratio: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Marginalize target-free context-only evidence for selected candidates.

    This deliberately excludes the RGB local density and every pose-dependent
    projection.  It is useful for train-only component supervision that asks
    the full RADIO/ALIKE context branch to distinguish a true support track
    from a coherent repeated-structure track on its own.
    """

    if not isinstance(runtime, CandidatePoseRGBSpatialRuntime):
        raise ValueError("selected context scoring requires a target-free runtime")
    missing = float(missing_edge_log_likelihood_ratio)
    if not math.isfinite(missing):
        raise ValueError("selected context scoring constant is invalid")
    device = prediction.context_log_likelihood_ratios.device
    active = runtime.to(device)
    points = torch.as_tensor(point_indices, dtype=torch.long, device=device).reshape(-1)
    candidates = torch.as_tensor(candidate_indices, dtype=torch.long, device=device).reshape(-1)
    if (
        len(points) == 0
        or candidates.shape != points.shape
        or torch.any(points < 0)
        or torch.any(points >= active.point_count)
        or torch.any(candidates < 0)
        or torch.any(candidates >= active.candidate_count)
        or prediction.context_log_likelihood_ratios.shape
        != active.support_image_indices.shape
        or prediction.context_edge_usable is None
        or prediction.context_edge_usable.shape != active.support_image_indices.shape
    ):
        raise ValueError("selected context scoring inputs are invalid")
    context = prediction.context_log_likelihood_ratios[points, candidates]
    usable = prediction.context_edge_usable[points, candidates]
    weights = active.candidate_view_weights[points, candidates]
    effective = torch.where(usable, context, torch.full_like(context, missing))
    safe_log_weights = torch.where(
        weights > 0.0,
        torch.log(weights),
        torch.full_like(weights, -torch.inf),
    )
    score = torch.logsumexp(effective + safe_log_weights, dim=1)
    candidate_usable = torch.any(usable & (weights > 0.0), dim=1)
    return score, candidate_usable


def context_candidate_logit_mixture(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidatePoseRGBSpatialEdgePrediction,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Marginalize target-free context evidence into one logit per candidate.

    This is the L0 identity path: it deliberately excludes frozen coarse
    priors, pose projections, RGB density, dustbin values, and target labels.
    It therefore answers only whether the query/support visual context makes a
    fixed candidate more plausible than its fixed peers.  The caller may join
    train-only identity labels *after* this function returns.
    """

    if not isinstance(runtime, CandidatePoseRGBSpatialRuntime):
        raise ValueError("context candidate mixture requires a target-free runtime")
    device = prediction.context_log_likelihood_ratios.device
    active = runtime.to(device)
    context = prediction.context_log_likelihood_ratios
    if prediction.context_edge_usable is None:
        raise ValueError("context candidate mixture lacks context source availability")
    usable = prediction.context_edge_usable & active.support_view_valid
    weights = active.candidate_view_weights
    if (
        context.shape != active.support_image_indices.shape
        or prediction.context_edge_usable.shape != context.shape
        or weights.shape != context.shape
    ):
        raise ValueError("context candidate mixture inputs are incompatible")
    active_edge = usable & (weights > 0.0)
    safe_log_weights = torch.where(
        active_edge,
        torch.log(weights.clamp_min(torch.finfo(weights.dtype).tiny)),
        torch.full_like(weights, -torch.inf),
    )
    logits = torch.logsumexp(context + safe_log_weights, dim=2)
    candidate_usable = torch.any(active_edge, dim=2)
    return logits, candidate_usable


def context_identity_cross_entropy_loss(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidatePoseRGBSpatialEdgePrediction,
    target_observed: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train strict observed-track identity against the fixed candidate set.

    ``target_observed`` is intentionally train-only.  Rows without a strict
    observed track are ignored rather than being silently relabelled as null:
    the top-L bank may still contain a geometrically valid, non-observed track.
    This prevents the identity head from learning a false null shortcut.
    """

    logits, candidate_usable = context_candidate_logit_mixture(
        runtime=runtime,
        prediction=prediction,
    )
    targets = torch.as_tensor(
        target_observed,
        dtype=torch.bool,
        device=logits.device,
    )
    if (
        targets.shape != logits.shape
        or torch.any(targets.sum(dim=1) > 1)
        or not torch.isfinite(logits[candidate_usable]).all()
    ):
        raise ValueError("context identity targets do not match candidate logits")
    labels = torch.argmax(targets.to(dtype=torch.long), dim=1)
    observed_rows = torch.any(targets, dim=1)
    label_usable = candidate_usable.gather(1, labels[:, None]).squeeze(1)
    competing_candidates = candidate_usable.sum(dim=1) >= 2
    active = observed_rows & label_usable & competing_candidates
    if not bool(active.any()):
        zero = prediction.context_log_likelihood_ratios.sum() * 0.0
        return zero, {
            "context_identity_active_rows": 0.0,
            "context_identity_top1_accuracy": 0.0,
            "context_identity_mean_margin": 0.0,
            "context_identity_cross_entropy": 0.0,
        }
    masked_logits = logits.masked_fill(~candidate_usable, -torch.inf)
    loss = F.cross_entropy(masked_logits[active], labels[active])
    selected = masked_logits[active].gather(1, labels[active, None]).squeeze(1)
    competing = masked_logits[active].clone()
    competing.scatter_(1, labels[active, None], -torch.inf)
    margin = selected - torch.amax(competing, dim=1)
    return loss, {
        "context_identity_active_rows": float(active.sum().item()),
        "context_identity_top1_accuracy": float(
            (torch.argmax(masked_logits[active], dim=1) == labels[active])
            .to(dtype=torch.float32)
            .mean()
            .detach()
            .item()
        ),
        "context_identity_mean_margin": float(margin.detach().mean().item()),
        "context_identity_cross_entropy": float(loss.detach().item()),
    }


def context_identity_support_permutation_margin_loss(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidatePoseRGBSpatialEdgePrediction,
    permuted_runtime: CandidatePoseRGBSpatialRuntime,
    permuted_prediction: CandidatePoseRGBSpatialEdgePrediction,
    target_observed: torch.Tensor,
    margin: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Require strict identity evidence to weaken under support derangement.

    The permutation preserves every query anchor, candidate slot, prior, and
    support-view weight.  It changes only the candidate-specific support image
    and coordinate appearance.  A positive gap is consequently visual evidence
    rather than a candidate-rank or geometry shortcut.
    """

    target_margin = float(margin)
    if not math.isfinite(target_margin) or target_margin < 0.0:
        raise ValueError("context identity permutation margin is invalid")
    normal_logits, normal_usable = context_candidate_logit_mixture(
        runtime=runtime,
        prediction=prediction,
    )
    permuted_logits, permuted_usable = context_candidate_logit_mixture(
        runtime=permuted_runtime,
        prediction=permuted_prediction,
    )
    targets = torch.as_tensor(target_observed, dtype=torch.bool, device=normal_logits.device)
    if (
        targets.shape != normal_logits.shape
        or permuted_logits.shape != normal_logits.shape
        or normal_usable.shape != normal_logits.shape
        or permuted_usable.shape != normal_logits.shape
        or torch.any(targets.sum(dim=1) > 1)
    ):
        raise ValueError("context identity permutation targets are invalid")
    labels = torch.argmax(targets.to(dtype=torch.long), dim=1)
    observed_rows = torch.any(targets, dim=1)
    normal_target_usable = normal_usable.gather(1, labels[:, None]).squeeze(1)
    permuted_target_usable = permuted_usable.gather(1, labels[:, None]).squeeze(1)
    active = observed_rows & normal_target_usable & permuted_target_usable
    if not bool(active.any()):
        zero = (
            prediction.context_log_likelihood_ratios.sum()
            + permuted_prediction.context_log_likelihood_ratios.sum()
        ) * 0.0
        return zero, {
            "context_identity_permutation_active_rows": 0.0,
            "context_identity_permutation_mean_gap": 0.0,
            "context_identity_permutation_win_fraction": 0.0,
            "context_identity_permutation_margin_loss": 0.0,
        }
    normal = normal_logits.gather(1, labels[:, None]).squeeze(1)
    permuted = permuted_logits.gather(1, labels[:, None]).squeeze(1)
    gaps = normal[active] - permuted[active]
    loss = F.softplus(
        torch.as_tensor(target_margin, dtype=gaps.dtype, device=gaps.device) - gaps
    ).mean()
    return loss, {
        "context_identity_permutation_active_rows": float(active.sum().item()),
        "context_identity_permutation_mean_gap": float(gaps.detach().mean().item()),
        "context_identity_permutation_win_fraction": float(
            (gaps.detach() > 0.0).to(dtype=torch.float32).mean().item()
        ),
        "context_identity_permutation_margin_loss": float(loss.detach().item()),
    }


@dataclass(frozen=True)
class CandidatePoseRGBSpatialScore:
    """Fixed-mixture pose score from target-free RGB spatial evidence."""

    pose_log_likelihood_ratios: torch.Tensor
    point_log_likelihood_ratios: torch.Tensor
    candidate_log_likelihood_ratios: torch.Tensor
    edge_log_likelihood_ratios: torch.Tensor
    edge_usable: torch.Tensor


def score_candidate_pose_rgb_spatial_batch(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidatePoseRGBSpatialEdgePrediction,
    candidate_projection_offsets_xy: torch.Tensor,
    candidate_projection_valid: torch.Tensor,
    missing_edge_log_likelihood_ratio: float = 0.0,
    max_abs_log_likelihood_ratio: float = 6.0,
) -> CandidatePoseRGBSpatialScore:
    """Score frozen top-L candidates under arbitrary caller-side poses."""

    if not isinstance(runtime, CandidatePoseRGBSpatialRuntime):
        raise ValueError("RGB spatial scoring requires a target-free runtime")
    device = prediction.joint_log_probabilities.device
    active = runtime.to(device)
    if prediction.joint_log_probabilities.shape[:3] != active.support_image_indices.shape:
        raise ValueError("RGB spatial prediction and runtime layouts differ")
    edge_llr, usable = edge_log_likelihood_ratio_at_pose_projection(
        prediction=prediction,
        candidate_projection_offsets_xy=candidate_projection_offsets_xy,
        candidate_projection_valid=candidate_projection_valid,
        missing_edge_log_likelihood_ratio=missing_edge_log_likelihood_ratio,
        max_abs_log_likelihood_ratio=max_abs_log_likelihood_ratio,
    )
    point_llr, candidate_llr = fixed_candidate_view_mixture_log_ratio(
        edge_log_likelihood_ratios=edge_llr,
        edge_usable=usable,
        candidate_view_weights=active.candidate_view_weights,
        candidate_probabilities=active.candidate_probabilities,
        null_probabilities=active.null_probabilities,
        missing_edge_log_likelihood_ratio=missing_edge_log_likelihood_ratio,
    )
    return CandidatePoseRGBSpatialScore(
        pose_log_likelihood_ratios=point_llr.mean(dim=1),
        point_log_likelihood_ratios=point_llr,
        candidate_log_likelihood_ratios=candidate_llr,
        edge_log_likelihood_ratios=edge_llr,
        edge_usable=usable,
    )


def spatial_density_nll(
    *,
    prediction: CandidatePoseRGBSpatialEdgePrediction,
    target_offsets_xy: torch.Tensor,
    target_dustbin: torch.Tensor,
    target_supervised: torch.Tensor | None = None,
    dustbin_weight: float = 1.0,
    balance_observed_and_dustbin: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train local density on train-only correct-pose target offsets.

    Dustbin is supervised here only as a conditional spatial-density target.
    It is deliberately excluded from pose-hypothesis scoring outside the local
    support window by :func:`edge_log_likelihood_ratio_at_pose_projection`.
    """

    weight = float(dustbin_weight)
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("dustbin weight must be finite and non-negative")
    targets = torch.as_tensor(
        target_offsets_xy,
        dtype=torch.float32,
        device=prediction.joint_log_probabilities.device,
    )
    dustbin = torch.as_tensor(
        target_dustbin,
        dtype=torch.bool,
        device=prediction.joint_log_probabilities.device,
    )
    if target_supervised is None:
        supervised = torch.ones_like(dustbin)
    else:
        supervised = torch.as_tensor(
            target_supervised,
            dtype=torch.bool,
            device=prediction.joint_log_probabilities.device,
        )
    if (
        targets.shape != (*prediction.joint_log_probabilities.shape[:2], 2)
        or dustbin.shape != targets.shape[:2]
        or supervised.shape != targets.shape[:2]
        or torch.any(dustbin & ~supervised)
    ):
        raise ValueError("spatial density targets do not match edge prediction")
    local_log_probability, in_window = continuous_joint_log_probability_at_offsets(
        joint_log_probabilities=prediction.joint_log_probabilities,
        offsets_xy=prediction.offsets_xy,
        query_offsets_xy=targets.unsqueeze(0),
    )
    dustbin_log_probability = prediction.joint_log_probabilities[..., -1].unsqueeze(0)
    target_log_probability = torch.where(
        dustbin.unsqueeze(0).unsqueeze(-1),
        dustbin_log_probability,
        local_log_probability,
    )
    if prediction.rgb_edge_usable is None or prediction.context_edge_usable is None:
        raise ValueError("spatial density requires source-specific availability")
    rgb_usable = prediction.rgb_edge_usable.unsqueeze(0)
    fusion_usable = (
        prediction.rgb_edge_usable & prediction.context_edge_usable
    ).unsqueeze(0)
    observed_usable = rgb_usable & in_window
    # Dustbin is emitted by the RGB/context fusion head. Do not report or
    # optimize a dustbin target when only one source is available; its
    # source-safe fallback is deliberately fixed-neutral.
    target_source_usable = torch.where(
        dustbin.unsqueeze(0).unsqueeze(-1), fusion_usable, observed_usable
    )
    target_usable = target_source_usable & supervised.unsqueeze(0).unsqueeze(-1)
    negative_log_probability = -target_log_probability
    observed = target_usable & ~dustbin.unsqueeze(0).unsqueeze(-1)
    dustbin_active = target_usable & dustbin.unsqueeze(0).unsqueeze(-1)
    if not bool(target_usable.any()):
        # Sparse registered identity supervision is intentionally absent for
        # many detector anchors.  Keep the pose objective trainable without
        # inventing a null target for those anchors.
        loss = prediction.spatial_logits.sum() * 0.0
    elif bool(balance_observed_and_dustbin):
        terms: list[torch.Tensor] = []
        if bool(observed.any()):
            terms.append(negative_log_probability[observed].mean())
        if bool(dustbin_active.any()):
            terms.append(weight * negative_log_probability[dustbin_active].mean())
        if not terms:
            loss = prediction.spatial_logits.sum() * 0.0
        else:
            # Preserve the configured dustbin relative weight without letting
            # its much larger candidate count dominate exact-identity rows.
            denominator = (1.0 if bool(observed.any()) else 0.0) + (
                weight if bool(dustbin_active.any()) else 0.0
            )
            loss = torch.stack(terms).sum() / max(denominator, 1e-12)
    else:
        per_edge_weight = torch.where(
            dustbin.unsqueeze(0).unsqueeze(-1),
            torch.full_like(target_log_probability, weight),
            torch.ones_like(target_log_probability),
        )
        loss = (negative_log_probability * per_edge_weight)[target_usable].mean()
    metrics = {
        "spatial_density_active_edges": float(target_usable.sum().item()),
        "spatial_density_observed_edges": float(observed.sum().item()),
        "spatial_density_dustbin_edges": float(dustbin_active.sum().item()),
        "spatial_density_supervised_edges": float(
            (
                torch.where(
                    dustbin.unsqueeze(0).unsqueeze(-1), fusion_usable, rgb_usable
                )
                & supervised.unsqueeze(0).unsqueeze(-1)
            ).sum().item()
        ),
        "spatial_density_mean_nll": float(loss.detach().item()),
    }
    return loss, metrics


def _crop_subpixel_grid_absolute_coordinates(
    *,
    image_sizes: torch.Tensor,
    image_indices: torch.Tensor,
    xy: torch.Tensor,
    grid_size: int,
    window_size: int,
) -> torch.Tensor:
    """Return normalized full-image coordinates for the sampled crop tokens.

    The crop sampler itself works in descriptor-grid coordinates.  This helper
    mirrors that conversion and keeps the token's image-frame phase explicit;
    it never receives support-track identity, pose, residual, or label.
    Invalid border tokens may have out-of-range coordinates but are masked by
    the paired visual crop before entering attention.
    """

    indices = torch.as_tensor(image_indices, dtype=torch.long, device=image_sizes.device).reshape(-1)
    coordinates = torch.as_tensor(xy, dtype=torch.float32, device=image_sizes.device)
    size = int(grid_size)
    width = int(window_size)
    if (
        image_sizes.ndim != 2
        or image_sizes.shape[1] != 2
        or coordinates.shape != (len(indices), 2)
        or len(indices) == 0
        or torch.any(indices < 0)
        or torch.any(indices >= len(image_sizes))
        or size < width
        or width < 1
        or width % 2 != 1
        or not torch.isfinite(coordinates).all()
    ):
        raise ValueError("absolute context crop coordinate inputs are invalid")
    selected_sizes = image_sizes.index_select(0, indices).to(dtype=torch.float32)
    extent = float(size - 1)
    center_x = coordinates[:, 0] / (selected_sizes[:, 0] - 1.0).clamp_min(1.0) * extent
    center_y = coordinates[:, 1] / (selected_sizes[:, 1] - 1.0).clamp_min(1.0) * extent
    radius = width // 2
    offsets = torch.arange(-radius, radius + 1, dtype=torch.float32, device=image_sizes.device)
    columns = center_x[:, None] + offsets[None, :]
    rows = center_y[:, None] + offsets[None, :]
    if size == 1:
        normalized_x = torch.zeros_like(columns)
        normalized_y = torch.zeros_like(rows)
    else:
        normalized_x = 2.0 * columns / extent - 1.0
        normalized_y = 2.0 * rows / extent - 1.0
    token_x = normalized_x[:, None, :].expand(-1, width, -1)
    token_y = normalized_y[:, :, None].expand(-1, -1, width)
    return torch.stack([token_x, token_y], dim=-1).reshape(len(indices), width * width, 2)


class _BidirectionalCrossAttentionContextCropEncoder(nn.Module):
    """Candidate-specific token interaction for one query/support crop pair.

    The previous context branch fused only same-index token products with a
    convolutional pool.  That is fast, but it cannot explicitly ask which
    support tokens explain a query token when a facade is shifted by one
    repeated cell.  This encoder keeps each crop two-dimensional through a
    self-attention and a bidirectional cross-attention pass before pooling a
    symmetric pair representation.  It receives neither global image position
    nor any target-side geometry; the positional code is only the fixed local
    crop grid shared by the two visual observations.
    """

    def __init__(
        self, descriptor_dim: int, hidden_dim: int, *, absolute_coordinates: bool = False
    ) -> None:
        super().__init__()
        dimension = int(hidden_dim)
        if int(descriptor_dim) <= 0 or dimension < 4:
            raise ValueError("cross-attention context encoder dimensions are invalid")
        self.query_projection = nn.Linear(int(descriptor_dim), dimension, bias=False)
        self.support_projection = nn.Linear(int(descriptor_dim), dimension, bias=False)
        self.position_projection = nn.Sequential(
            nn.Linear(5, dimension),
            nn.GELU(),
            nn.Linear(dimension, dimension),
        )
        self.absolute_coordinates = bool(absolute_coordinates)
        self.query_absolute_position_projection = (
            nn.Sequential(
                nn.Linear(5, dimension),
                nn.GELU(),
                nn.Linear(dimension, dimension),
            )
            if self.absolute_coordinates
            else None
        )
        self.support_absolute_position_projection = (
            nn.Sequential(
                nn.Linear(5, dimension),
                nn.GELU(),
                nn.Linear(dimension, dimension),
            )
            if self.absolute_coordinates
            else None
        )
        self.query_self_query = nn.Linear(dimension, dimension, bias=False)
        self.query_self_key = nn.Linear(dimension, dimension, bias=False)
        self.query_self_value = nn.Linear(dimension, dimension, bias=False)
        self.support_self_query = nn.Linear(dimension, dimension, bias=False)
        self.support_self_key = nn.Linear(dimension, dimension, bias=False)
        self.support_self_value = nn.Linear(dimension, dimension, bias=False)
        self.query_cross_query = nn.Linear(dimension, dimension, bias=False)
        self.query_cross_key = nn.Linear(dimension, dimension, bias=False)
        self.query_cross_value = nn.Linear(dimension, dimension, bias=False)
        self.support_cross_query = nn.Linear(dimension, dimension, bias=False)
        self.support_cross_key = nn.Linear(dimension, dimension, bias=False)
        self.support_cross_value = nn.Linear(dimension, dimension, bias=False)
        self.query_self_norm = nn.LayerNorm(dimension)
        self.support_self_norm = nn.LayerNorm(dimension)
        self.query_cross_norm = nn.LayerNorm(dimension)
        self.support_cross_norm = nn.LayerNorm(dimension)
        self.output = nn.Sequential(
            nn.LayerNorm(dimension * 6),
            nn.Linear(dimension * 6, dimension * 2),
            nn.GELU(),
            nn.Linear(dimension * 2, dimension),
            nn.GELU(),
        )
        self._scale = float(dimension) ** -0.5

    @staticmethod
    def _masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        mask = valid.to(dtype=values.dtype).unsqueeze(-1)
        return (values * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    @staticmethod
    def _masked_max(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        masked = values.masked_fill(~valid.unsqueeze(-1), torch.finfo(values.dtype).min)
        return masked.amax(dim=1)

    def _attention(
        self,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_valid: torch.Tensor,
        key_valid: torch.Tensor,
    ) -> torch.Tensor:
        if (
            query.ndim != 3
            or key.shape != value.shape
            or query.shape[0] != key.shape[0]
            or query.shape[2] != key.shape[2]
            or query_valid.shape != query.shape[:2]
            or key_valid.shape != key.shape[:2]
            or not bool(torch.all(torch.any(key_valid, dim=1)))
        ):
            raise ValueError("cross-attention context token masks are invalid")
        logits = torch.matmul(query, key.transpose(1, 2)) * self._scale
        logits = logits.masked_fill(~key_valid.unsqueeze(1), -1.0e4)
        weights = torch.softmax(logits, dim=2)
        attended = torch.matmul(weights, value)
        return attended * query_valid.unsqueeze(-1).to(dtype=attended.dtype)

    @staticmethod
    def _local_position_features(
        *, window_size: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        width = int(window_size)
        values = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        x, y = torch.meshgrid(values, values, indexing="xy")
        return torch.stack([x, y, x * y, x.square(), y.square()], dim=-1).reshape(-1, 5)

    @staticmethod
    def _absolute_position_features(coordinates: torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(coordinates, dtype=torch.float32)
        if values.ndim != 3 or values.shape[2] != 2 or not torch.isfinite(values).all():
            raise ValueError("absolute context coordinates are invalid")
        x = values[..., 0]
        y = values[..., 1]
        return torch.stack([x, y, x * y, x.square(), y.square()], dim=-1)

    def forward(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
        query_valid: torch.Tensor,
        support_valid: torch.Tensor,
        *,
        window_size: int,
        query_absolute_coordinates: torch.Tensor | None = None,
        support_absolute_coordinates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            query_tokens.ndim != 3
            or support_tokens.shape != query_tokens.shape
            or query_tokens.shape[1] != int(window_size) ** 2
            or query_valid.shape != query_tokens.shape[:2]
            or support_valid.shape != query_tokens.shape[:2]
        ):
            raise ValueError("cross-attention context crop encoder inputs are invalid")
        query_mask = torch.as_tensor(
            query_valid, dtype=torch.bool, device=query_tokens.device
        )
        support_mask = torch.as_tensor(
            support_valid, dtype=torch.bool, device=query_tokens.device
        )
        if not bool(torch.all(torch.any(query_mask, dim=1))) or not bool(
            torch.all(torch.any(support_mask, dim=1))
        ):
            raise ValueError("cross-attention context crop encoder received an empty crop")
        if self.absolute_coordinates:
            if (
                query_absolute_coordinates is None
                or support_absolute_coordinates is None
                or torch.as_tensor(query_absolute_coordinates).shape
                != query_tokens.shape[:2] + (2,)
                or torch.as_tensor(support_absolute_coordinates).shape
                != query_tokens.shape[:2] + (2,)
                or self.query_absolute_position_projection is None
                or self.support_absolute_position_projection is None
            ):
                raise ValueError("absolute context crop encoder coordinates are incompatible")
        elif query_absolute_coordinates is not None or support_absolute_coordinates is not None:
            raise ValueError("relative context crop encoder received absolute coordinates")
        position = self.position_projection(
            self._local_position_features(
                window_size=int(window_size),
                device=query_tokens.device,
                dtype=torch.float32,
            )
        ).unsqueeze(0)
        query = self.query_projection(query_tokens.to(dtype=torch.float32)) + position
        support = self.support_projection(support_tokens.to(dtype=torch.float32)) + position
        if self.absolute_coordinates:
            assert self.query_absolute_position_projection is not None
            assert self.support_absolute_position_projection is not None
            query = query + self.query_absolute_position_projection(
                self._absolute_position_features(query_absolute_coordinates).to(
                    device=query.device, dtype=query.dtype
                )
            )
            support = support + self.support_absolute_position_projection(
                self._absolute_position_features(support_absolute_coordinates).to(
                    device=support.device, dtype=support.dtype
                )
            )
        query = query * query_mask.unsqueeze(-1).to(dtype=query.dtype)
        support = support * support_mask.unsqueeze(-1).to(dtype=support.dtype)
        query_self = self.query_self_norm(
            query
            + self._attention(
                query=self.query_self_query(query),
                key=self.query_self_key(query),
                value=self.query_self_value(query),
                query_valid=query_mask,
                key_valid=query_mask,
            )
        )
        support_self = self.support_self_norm(
            support
            + self._attention(
                query=self.support_self_query(support),
                key=self.support_self_key(support),
                value=self.support_self_value(support),
                query_valid=support_mask,
                key_valid=support_mask,
            )
        )
        query_cross = self.query_cross_norm(
            query_self
            + self._attention(
                query=self.query_cross_query(query_self),
                key=self.query_cross_key(support_self),
                value=self.query_cross_value(support_self),
                query_valid=query_mask,
                key_valid=support_mask,
            )
        )
        support_cross = self.support_cross_norm(
            support_self
            + self._attention(
                query=self.support_cross_query(support_self),
                key=self.support_cross_key(query_self),
                value=self.support_cross_value(query_self),
                query_valid=support_mask,
                key_valid=query_mask,
            )
        )
        query_mean = self._masked_mean(query_cross, query_mask)
        support_mean = self._masked_mean(support_cross, support_mask)
        query_max = self._masked_max(query_cross, query_mask)
        support_max = self._masked_max(support_cross, support_mask)
        pair = torch.cat(
            [
                query_mean,
                support_mean,
                query_mean * support_mean,
                torch.abs(query_mean - support_mean),
                query_max * support_max,
                torch.abs(query_max - support_max),
            ],
            dim=1,
        )
        return self.output(pair)


class CandidatePoseRGBSpatialLikelihood(nn.Module):
    """High-resolution local RGB density plus broad candidate context LLR.

    :meth:`forward` intentionally accepts no pose projection, residual, target,
    track ID, candidate rank, or coarse score.  It returns all target-free edge
    densities once; callers may then score several hypothesis projections
    without rerunning the expensive visual encoder.
    """

    def __init__(
        self,
        *,
        sources: Mapping[str, torch.Tensor] | None,
        image_sizes: torch.Tensor,
        search_radius_px: float = 8.0,
        context_radius_px: float = 12.0,
        step_px: float = 1.0,
        texture_feature_dim: int = 32,
        hidden_dim: int = 32,
        max_abs_context_log_ratio: float = 3.0,
        edge_chunk_size: int = 256,
        activation_checkpointing: bool = False,
        rgb_temperature: float = 10.0,
        context_windows: Mapping[str, int] | None = None,
        context_encoder_arch: str = "conv_v1",
        context_source_dimensions: Mapping[str, int] | None = None,
    ) -> None:
        super().__init__()
        source_grids = None if sources is None else dict(sources)
        if source_grids is not None and set(source_grids) != set(
            CANDIDATE_POSE_RGB_SPATIAL_CONTEXT_WINDOWS
        ):
            raise ValueError("candidate RGB spatial context source set is incomplete")
        if source_grids is None:
            if context_source_dimensions is None or set(context_source_dimensions) != set(
                CANDIDATE_POSE_RGB_SPATIAL_CONTEXT_WINDOWS
            ):
                raise ValueError(
                    "source-free RGB-only construction requires all context source dimensions"
                )
            source_dimensions = {
                str(name): int(value) for name, value in context_source_dimensions.items()
            }
            if any(value <= 0 for value in source_dimensions.values()):
                raise ValueError("candidate RGB spatial context source dimensions are invalid")
        else:
            source_dimensions = {}
        radius = float(search_radius_px)
        context = float(context_radius_px)
        step = float(step_px)
        if (
            not all(math.isfinite(value) for value in (radius, context, step, float(rgb_temperature)))
            or radius < step
            or context <= 0.0
            or step <= 0.0
            or int(texture_feature_dim) <= 0
            or int(hidden_dim) < 4
            or int(edge_chunk_size) <= 0
            or float(max_abs_context_log_ratio) <= 0.0
        ):
            raise ValueError("candidate RGB spatial likelihood configuration is invalid")
        grid_side = int(round((2.0 * radius) / step)) + 1
        if grid_side < 3 or grid_side % 2 != 1 or not math.isclose(
            (grid_side - 1) * step, 2.0 * radius, rel_tol=1e-5, abs_tol=1e-5
        ):
            raise ValueError("search radius must form an odd regular local grid")
        patch_side = int(round((2.0 * (radius + context)) / step)) + 1
        if patch_side <= grid_side or not math.isclose(
            (patch_side - 1) * step, 2.0 * (radius + context), rel_tol=1e-5, abs_tol=1e-5
        ):
            raise ValueError("RGB patch geometry is invalid")
        sizes = torch.as_tensor(image_sizes, dtype=torch.float32)
        if sizes.ndim != 2 or sizes.shape[1] != 2 or torch.any(sizes <= 1.0):
            raise ValueError("candidate RGB spatial image sizes are invalid")
        resolved_context_windows = resolve_candidate_pose_rgb_spatial_context_windows(
            context_windows
        )
        resolved_context_encoder_arch = resolve_candidate_pose_rgb_spatial_context_encoder_arch(
            context_encoder_arch
        )
        encoders: dict[str, nn.Module] = {}
        for name, window in resolved_context_windows.items():
            if source_grids is not None:
                grid = torch.as_tensor(source_grids[name], dtype=torch.float32)
                if (
                    grid.ndim != 4
                    or grid.shape[0] != len(sizes)
                    or grid.shape[1] != grid.shape[2]
                    or grid.shape[1] < int(window)
                    or grid.shape[3] <= 0
                    or not torch.isfinite(grid).all()
                ):
                    raise ValueError(f"candidate RGB spatial {name} source grid is invalid")
                norms = torch.linalg.vector_norm(grid, dim=-1)
                if torch.max(torch.abs(norms - 1.0)) > 5e-3:
                    raise ValueError(f"candidate RGB spatial {name} descriptors are not normalized")
                source_dimensions[name] = int(grid.shape[3])
                self.register_buffer(f"_{name}_grid", grid, persistent=False)
            if resolved_context_encoder_arch == "conv_v1":
                encoders[name] = _FullContextCropEncoder(
                    int(source_dimensions[name]), int(hidden_dim)
                )
            elif resolved_context_encoder_arch == "cross_attention_v2":
                encoders[name] = _BidirectionalCrossAttentionContextCropEncoder(
                    int(source_dimensions[name]), int(hidden_dim)
                )
            else:
                encoders[name] = _BidirectionalCrossAttentionContextCropEncoder(
                    int(source_dimensions[name]), int(hidden_dim), absolute_coordinates=True
                )
        self.context_encoders = nn.ModuleDict(encoders)
        self.context_windows = resolved_context_windows
        self.context_encoder_arch = resolved_context_encoder_arch
        self.context_sources_available = source_grids is not None
        self.register_buffer("_image_sizes", sizes, persistent=False)
        offsets = local_offset_grid(
            search_radius_px=radius, step_px=step, device=None, dtype=torch.float32
        )
        self.register_buffer("_offsets_xy", offsets, persistent=False)
        self.texture_encoder = TexturePatchEncoder(
            feature_dim=int(texture_feature_dim),
            hidden_dim=max(int(texture_feature_dim), int(hidden_dim)),
            input_mode="rgb_graygrad",
            encoder_arch="fpn",
        )
        context_feature_dim = int(hidden_dim) * len(encoders) + 9
        density_feature_dim = context_feature_dim + 6
        self.context_identity_head = nn.Sequential(
            nn.LayerNorm(context_feature_dim),
            nn.Linear(context_feature_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )
        self.spatial_residual_head = nn.Sequential(
            nn.LayerNorm(density_feature_dim),
            nn.Linear(density_feature_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(len(offsets))),
        )
        self.non_dustbin_head = nn.Sequential(
            nn.LayerNorm(density_feature_dim),
            nn.Linear(density_feature_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )
        for head in (self.spatial_residual_head, self.non_dustbin_head):
            final = head[-1]
            assert isinstance(final, nn.Linear)
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        context_final = self.context_identity_head[-1]
        assert isinstance(context_final, nn.Linear)
        nn.init.normal_(
            context_final.weight,
            mean=0.0,
            std=float(CONTEXT_IDENTITY_HEAD_FINAL_WEIGHT_STD),
        )
        nn.init.zeros_(context_final.bias)
        self.search_radius_px = radius
        self.context_radius_px = context
        self.step_px = step
        self.patch_side = patch_side
        self.max_abs_context_log_ratio = float(max_abs_context_log_ratio)
        self.edge_chunk_size = int(edge_chunk_size)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.rgb_temperature = float(rgb_temperature)

    @property
    def device(self) -> torch.device:
        return self._image_sizes.device

    def _validate_runtime_indices(self, runtime: CandidatePoseRGBSpatialRuntime) -> None:
        image_count = len(self._image_sizes)
        if (
            torch.any(runtime.query_image_indices >= image_count)
            or torch.any(runtime.support_image_indices >= image_count)
        ):
            raise ValueError("candidate RGB spatial runtime image index is out of range")

    def _encode_context_chunk(
        self,
        *,
        query_image_indices: torch.Tensor,
        query_xy: torch.Tensor,
        support_image_indices: torch.Tensor,
        support_xy: torch.Tensor,
        context_appearance_mode: str = "visual",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.context_sources_available:
            raise RuntimeError(
                "candidate RGB spatial context grids were intentionally omitted for RGB-only execution"
            )
        appearance_mode = str(context_appearance_mode).strip().lower()
        if appearance_mode not in {"visual", "position_only"}:
            raise ValueError("candidate context appearance mode is invalid")
        if appearance_mode == "position_only" and self.context_encoder_arch != "absolute_cross_attention_v3":
            raise ValueError("position-only context control requires absolute_cross_attention_v3")
        encoded_scales: list[torch.Tensor] = []
        alike_correlations: torch.Tensor | None = None
        all_valid = torch.ones((len(query_xy),), dtype=torch.bool, device=self.device)
        for name, window in self.context_windows.items():
            grid = getattr(self, f"_{name}_grid")
            query_crop, query_valid = _crop_subpixel_grid_tokens(
                image_grids=grid,
                image_sizes=self._image_sizes,
                image_indices=query_image_indices,
                xy=query_xy,
                window_size=int(window),
            )
            support_crop, support_valid = _crop_subpixel_grid_tokens(
                image_grids=grid,
                image_sizes=self._image_sizes,
                image_indices=support_image_indices,
                xy=support_xy,
                window_size=int(window),
            )
            paired = torch.any(query_valid & support_valid, dim=1)
            all_valid &= paired
            # Let the encoder keep a uniform batch shape while explicitly
            # removing empty border crops from both score and gradient paths.
            safe_query_valid = query_valid.clone()
            safe_support_valid = support_valid.clone()
            empty = ~paired
            safe_query_valid[empty, 0] = True
            safe_support_valid[empty, 0] = True
            if appearance_mode == "position_only":
                query_crop = torch.zeros_like(query_crop)
                support_crop = torch.zeros_like(support_crop)
            encoder = self.context_encoders[name]
            if self.context_encoder_arch == "absolute_cross_attention_v3":
                query_absolute = _crop_subpixel_grid_absolute_coordinates(
                    image_sizes=self._image_sizes,
                    image_indices=query_image_indices,
                    xy=query_xy,
                    grid_size=int(grid.shape[1]),
                    window_size=int(window),
                )
                support_absolute = _crop_subpixel_grid_absolute_coordinates(
                    image_sizes=self._image_sizes,
                    image_indices=support_image_indices,
                    xy=support_xy,
                    grid_size=int(grid.shape[1]),
                    window_size=int(window),
                )
                encoded_scales.append(
                    encoder(
                        query_crop,
                        support_crop,
                        safe_query_valid,
                        safe_support_valid,
                        window_size=int(window),
                        query_absolute_coordinates=query_absolute,
                        support_absolute_coordinates=support_absolute,
                    )
                )
            else:
                encoded_scales.append(
                    encoder(
                        query_crop,
                        support_crop,
                        safe_query_valid,
                        safe_support_valid,
                        window_size=int(window),
                    )
                )
            if name == "alike":
                alike_correlations = _alike_shift_correlations(
                    query_crop,
                    support_crop,
                    window_size=int(window),
                    query_valid=query_valid,
                    support_valid=support_valid,
                )
        if alike_correlations is None:
            raise RuntimeError("candidate RGB spatial ALIKE context branch is missing")
        return torch.cat([*encoded_scales, alike_correlations], dim=1), all_valid

    def _edge_chunk(
        self,
        *,
        query_image_indices: torch.Tensor,
        query_xy: torch.Tensor,
        support_image_indices: torch.Tensor,
        support_xy: torch.Tensor,
        query_texture_features: torch.Tensor,
        support_rgb_patches: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        context_features, context_valid = self._encode_context_chunk(
            query_image_indices=query_image_indices,
            query_xy=query_xy,
            support_image_indices=support_image_indices,
            support_xy=support_xy,
        )
        if (
            query_texture_features.ndim != 4
            or query_texture_features.shape[0] != len(query_xy)
            or not torch.isfinite(query_texture_features).all()
        ):
            raise ValueError("candidate RGB spatial query texture features are invalid")
        support_features = self.texture_encoder(support_rgb_patches)
        raw_spatial, offsets = template_search_cost_volume_logits(
            query_texture_features,
            support_features,
            search_radius_px=self.search_radius_px,
            context_radius_px=self.context_radius_px,
            step_px=self.step_px,
            temperature=self.rgb_temperature,
        )
        if not torch.allclose(
            offsets.to(device=self.device, dtype=torch.float32), self._offsets_xy,
            atol=1e-5,
            rtol=1e-5,
        ):
            raise RuntimeError("RGB cost volume emitted an unexpected local offset grid")
        quality = cost_volume_quality_features(raw_spatial)
        density_features = torch.cat([context_features, quality], dim=1)
        residual = self.spatial_residual_head(density_features)
        non_dustbin = self.non_dustbin_head(density_features).reshape(-1)
        context_llr = bounded_log_likelihood_ratio(
            self.context_identity_head(context_features).reshape(-1),
            max_abs_log_ratio=self.max_abs_context_log_ratio,
        )
        return raw_spatial, residual, non_dustbin, context_llr, context_valid

    def _edge_chunk_from_tensors(
        self,
        query_image_indices: torch.Tensor,
        query_xy: torch.Tensor,
        support_image_indices: torch.Tensor,
        support_xy: torch.Tensor,
        query_texture_features: torch.Tensor,
        support_rgb_patches: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._edge_chunk(
            query_image_indices=query_image_indices,
            query_xy=query_xy,
            support_image_indices=support_image_indices,
            support_xy=support_xy,
            query_texture_features=query_texture_features,
            support_rgb_patches=support_rgb_patches,
        )

    def _rgb_cost_volume_chunk(
        self,
        *,
        query_texture_features: torch.Tensor,
        support_rgb_patches: torch.Tensor,
    ) -> torch.Tensor:
        """Compute only candidate-specific RGB spatial evidence for one edge chunk.

        This deliberately excludes RADIO/ALIKE context, the learned residual
        grid, and the learned dustbin.  It is the narrow experiment boundary
        used when testing whether raw high-resolution appearance can resolve a
        correct pose from a coherent repeat on its own.
        """

        if (
            query_texture_features.ndim != 4
            or query_texture_features.shape[0] != support_rgb_patches.shape[0]
            or support_rgb_patches.ndim != 4
            or support_rgb_patches.shape[1:] != (3, self.patch_side, self.patch_side)
            or not torch.isfinite(query_texture_features).all()
            or not torch.isfinite(support_rgb_patches).all()
        ):
            raise ValueError("candidate RGB-only cost-volume chunk inputs are invalid")
        support_features = self.texture_encoder(support_rgb_patches)
        raw_spatial, offsets = template_search_cost_volume_logits(
            query_texture_features,
            support_features,
            search_radius_px=self.search_radius_px,
            context_radius_px=self.context_radius_px,
            step_px=self.step_px,
            temperature=self.rgb_temperature,
        )
        if not torch.allclose(
            offsets.to(device=self.device, dtype=torch.float32),
            self._offsets_xy,
            atol=1e-5,
            rtol=1e-5,
        ):
            raise RuntimeError("RGB-only cost volume emitted an unexpected local offset grid")
        return raw_spatial

    def _rgb_cost_volume_chunk_from_tensors(
        self,
        query_texture_features: torch.Tensor,
        support_rgb_patches: torch.Tensor,
    ) -> torch.Tensor:
        """Checkpointable tensor-only wrapper for RGB-only edge scoring."""

        return self._rgb_cost_volume_chunk(
            query_texture_features=query_texture_features,
            support_rgb_patches=support_rgb_patches,
        )

    def _rgb_window_usable(
        self,
        *,
        image_indices: torch.Tensor,
        centers_xy: torch.Tensor,
    ) -> torch.Tensor:
        """Reject padded RGB windows instead of letting border pixels repeat.

        RGB crops use ``padding_mode='border'`` for numerically stable sampling.
        That is useful for generic measurement, but a candidate likelihood must
        not regard duplicated border pixels as evidence.  The runtime encoder
        therefore marks such edges unusable before any pose projection is
        scored.
        """

        indices = torch.as_tensor(image_indices, dtype=torch.long, device=self.device).reshape(-1)
        centers = torch.as_tensor(centers_xy, dtype=torch.float32, device=self.device)
        if (
            centers.shape != (len(indices), 2)
            or len(indices) == 0
            or torch.any(indices < 0)
            or torch.any(indices >= len(self._image_sizes))
            or not torch.isfinite(centers).all()
        ):
            raise ValueError("RGB window ownership or centers are invalid")
        sizes = self._image_sizes.index_select(0, indices).to(dtype=torch.float32)
        radius = float(self.search_radius_px + self.context_radius_px)
        return (
            (centers[:, 0] >= radius)
            & (centers[:, 1] >= radius)
            & (centers[:, 0] <= sizes[:, 0] - 1.0 - radius)
            & (centers[:, 1] <= sizes[:, 1] - 1.0 - radius)
        )

    def _rgb_cost_volume_only_prediction(
        self,
        *,
        runtime: CandidatePoseRGBSpatialRuntime,
        query_rgb_patches: torch.Tensor,
        support_rgb_patches: torch.Tensor,
    ) -> CandidatePoseRGBSpatialEdgePrediction:
        """Emit a high-resolution RGB-only target-free local density.

        The result is intentionally not a shortcut version of the combined
        scorer: all context and learned scalar/grid heads are structurally
        bypassed.  Its only trainable path is the query/support RGB FPN followed
        by the candidate-specific template cost volume.
        """

        active = runtime.to(self.device)
        self._validate_runtime_indices(active)
        point_count = active.point_count
        candidate_count = active.candidate_count
        view_count = active.support_view_count
        query_patches = torch.as_tensor(
            query_rgb_patches, dtype=torch.float32, device=self.device
        )
        support_patches = torch.as_tensor(
            support_rgb_patches, dtype=torch.float32, device=self.device
        )
        expected_query_shape = (point_count, 3, self.patch_side, self.patch_side)
        expected_support_shape = (
            point_count,
            candidate_count,
            view_count,
            3,
            self.patch_side,
            self.patch_side,
        )
        if (
            query_patches.shape != expected_query_shape
            or support_patches.shape != expected_support_shape
            or not torch.isfinite(query_patches).all()
            or not torch.isfinite(support_patches).all()
        ):
            raise ValueError("candidate RGB-only patches do not match the fixed layout")

        edge_count = point_count * candidate_count * view_count
        query_texture_features = self.texture_encoder(query_patches)
        edge_to_point = torch.arange(point_count, device=self.device).repeat_interleave(
            candidate_count * view_count
        )
        flat_support_patches = support_patches.reshape(
            edge_count, 3, self.patch_side, self.patch_side
        )
        spatial_parts: list[torch.Tensor] = []
        for begin in range(0, edge_count, self.edge_chunk_size):
            end = min(begin + self.edge_chunk_size, edge_count)
            tensors = (
                query_texture_features[edge_to_point[begin:end]],
                flat_support_patches[begin:end],
            )
            if self.training and self.activation_checkpointing and torch.is_grad_enabled():
                raw_spatial = checkpoint(
                    self._rgb_cost_volume_chunk_from_tensors, *tensors, use_reentrant=False
                )
            else:
                raw_spatial = self._rgb_cost_volume_chunk(
                    query_texture_features=tensors[0], support_rgb_patches=tensors[1]
                )
            spatial_parts.append(raw_spatial)
        spatial_logits = torch.cat(spatial_parts, dim=0).reshape(
            point_count, candidate_count, view_count, -1
        )
        zero_edge = torch.zeros(
            (point_count, candidate_count, view_count),
            dtype=spatial_logits.dtype,
            device=self.device,
        )
        joint = normalized_spatial_log_probabilities_with_dustbin(
            spatial_logits.reshape(-1, spatial_logits.shape[-1]), zero_edge.reshape(-1)
        ).reshape(point_count, candidate_count, view_count, -1)
        query_usable = self._rgb_window_usable(
            image_indices=active.query_image_indices, centers_xy=active.query_xy
        )
        support_usable = self._rgb_window_usable(
            image_indices=active.support_image_indices.reshape(-1),
            centers_xy=active.support_xy.reshape(-1, 2),
        ).reshape(point_count, candidate_count, view_count)
        edge_usable = (
            active.support_view_valid
            & query_usable[:, None, None]
            & support_usable
        )
        return CandidatePoseRGBSpatialEdgePrediction(
            spatial_logits=spatial_logits,
            non_dustbin_logits=zero_edge,
            joint_log_probabilities=joint,
            offsets_xy=self._offsets_xy,
            context_log_likelihood_ratios=zero_edge,
            edge_usable=edge_usable,
            raw_spatial_logits=spatial_logits,
            spatial_residual_logits=torch.zeros_like(spatial_logits),
            rgb_edge_usable=edge_usable,
            context_edge_usable=torch.zeros_like(edge_usable),
        )

    def _context_chunk_from_tensors(
        self,
        query_image_indices: torch.Tensor,
        query_xy: torch.Tensor,
        support_image_indices: torch.Tensor,
        support_xy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Checkpointable tensor-only wrapper for the L0 context path."""

        return self._encode_context_chunk(
            query_image_indices=query_image_indices,
            query_xy=query_xy,
            support_image_indices=support_image_indices,
            support_xy=support_xy,
        )

    def _context_only_prediction(
        self,
        *,
        runtime: CandidatePoseRGBSpatialRuntime,
        context_appearance_mode: str = "visual",
    ) -> CandidatePoseRGBSpatialEdgePrediction:
        """Emit L0 identity context without invoking the RGB spatial branch."""

        active = runtime.to(self.device)
        self._validate_runtime_indices(active)
        point_count = active.point_count
        candidate_count = active.candidate_count
        view_count = active.support_view_count
        edge_count = point_count * candidate_count * view_count
        flat_query_indices = active.query_image_indices[:, None, None].expand(
            -1, candidate_count, view_count
        ).reshape(-1)
        flat_query_xy = active.query_xy[:, None, None, :].expand(
            -1, candidate_count, view_count, -1
        ).reshape(-1, 2)
        flat_support_indices = active.support_image_indices.reshape(-1)
        flat_support_xy = active.support_xy.reshape(-1, 2)
        context_parts: list[torch.Tensor] = []
        crop_valid_parts: list[torch.Tensor] = []
        for begin in range(0, edge_count, self.edge_chunk_size):
            end = min(begin + self.edge_chunk_size, edge_count)
            tensors = (
                flat_query_indices[begin:end],
                flat_query_xy[begin:end],
                flat_support_indices[begin:end],
                flat_support_xy[begin:end],
            )
            if (
                self.training
                and self.activation_checkpointing
                and torch.is_grad_enabled()
                and str(context_appearance_mode).strip().lower() == "visual"
            ):
                context_features, crop_valid = checkpoint(
                    self._context_chunk_from_tensors, *tensors, use_reentrant=False
                )
            else:
                context_features, crop_valid = self._encode_context_chunk(
                    query_image_indices=tensors[0],
                    query_xy=tensors[1],
                    support_image_indices=tensors[2],
                    support_xy=tensors[3],
                    context_appearance_mode=context_appearance_mode,
                )
            context_parts.append(
                bounded_log_likelihood_ratio(
                    self.context_identity_head(context_features).reshape(-1),
                    max_abs_log_ratio=self.max_abs_context_log_ratio,
                )
            )
            crop_valid_parts.append(crop_valid)
        context_llr = torch.cat(context_parts, dim=0).reshape(
            point_count, candidate_count, view_count
        )
        crop_valid = torch.cat(crop_valid_parts, dim=0).reshape(
            point_count, candidate_count, view_count
        )
        spatial_logits = torch.zeros(
            (point_count, candidate_count, view_count, len(self._offsets_xy)),
            dtype=torch.float32,
            device=self.device,
        )
        non_dustbin_logits = torch.zeros(
            (point_count, candidate_count, view_count),
            dtype=torch.float32,
            device=self.device,
        )
        joint = normalized_spatial_log_probabilities_with_dustbin(
            spatial_logits.reshape(-1, spatial_logits.shape[-1]),
            non_dustbin_logits.reshape(-1),
        ).reshape(point_count, candidate_count, view_count, -1)
        context_edge_usable = crop_valid & active.support_view_valid
        return CandidatePoseRGBSpatialEdgePrediction(
            spatial_logits=spatial_logits,
            non_dustbin_logits=non_dustbin_logits,
            joint_log_probabilities=joint,
            offsets_xy=self._offsets_xy,
            context_log_likelihood_ratios=context_llr,
            edge_usable=context_edge_usable,
            raw_spatial_logits=torch.zeros_like(spatial_logits),
            spatial_residual_logits=torch.zeros_like(spatial_logits),
            rgb_edge_usable=torch.zeros_like(context_edge_usable),
            context_edge_usable=context_edge_usable,
        )

    def forward_context_only(
        self,
        *,
        runtime: CandidatePoseRGBSpatialRuntime,
        context_appearance_mode: str = "visual",
    ) -> CandidatePoseRGBSpatialEdgePrediction:
        """Public L0 identity-only forward path for non-DDP callers."""

        if not isinstance(runtime, CandidatePoseRGBSpatialRuntime):
            raise ValueError("candidate RGB spatial model requires a target-free runtime")
        return self._context_only_prediction(
            runtime=runtime, context_appearance_mode=context_appearance_mode
        )

    def forward(
        self,
        *,
        runtime: CandidatePoseRGBSpatialRuntime,
        query_rgb_patches: torch.Tensor | None = None,
        support_rgb_patches: torch.Tensor | None = None,
        context_only: bool = False,
        rgb_cost_volume_only: bool = False,
        context_appearance_mode: str = "visual",
    ) -> CandidatePoseRGBSpatialEdgePrediction:
        """Encode every fixed edge once, without any pose-hypothesis input.

        ``context_only`` is the explicit L0 path.  It accepts no RGB patch and
        cannot accidentally train or evaluate the local spatial density.
        ``rgb_cost_volume_only`` is the complementary high-resolution path: it
        bypasses every RADIO/ALIKE context and learned residual/dustbin head so
        a train-only experiment can attribute its result solely to real RGB
        candidate-specific local appearance.
        """

        if not isinstance(runtime, CandidatePoseRGBSpatialRuntime):
            raise ValueError("candidate RGB spatial model requires a target-free runtime")
        if bool(context_only) and bool(rgb_cost_volume_only):
            raise ValueError("context-only and RGB-only candidate likelihood modes are exclusive")
        if bool(context_only):
            if query_rgb_patches is not None or support_rgb_patches is not None:
                raise ValueError("context-only candidate likelihood must not receive RGB patches")
            return self._context_only_prediction(
                runtime=runtime, context_appearance_mode=context_appearance_mode
            )
        if query_rgb_patches is None or support_rgb_patches is None:
            raise ValueError("full candidate likelihood requires query and support RGB patches")
        if str(context_appearance_mode).strip().lower() != "visual":
            raise ValueError("full candidate likelihood requires visual context appearance")
        if bool(rgb_cost_volume_only):
            return self._rgb_cost_volume_only_prediction(
                runtime=runtime,
                query_rgb_patches=query_rgb_patches,
                support_rgb_patches=support_rgb_patches,
            )
        active = runtime.to(self.device)
        self._validate_runtime_indices(active)
        point_count = active.point_count
        candidate_count = active.candidate_count
        view_count = active.support_view_count
        query_patches = torch.as_tensor(
            query_rgb_patches, dtype=torch.float32, device=self.device
        )
        support_patches = torch.as_tensor(
            support_rgb_patches, dtype=torch.float32, device=self.device
        )
        expected_query_shape = (point_count, 3, self.patch_side, self.patch_side)
        expected_support_shape = (
            point_count,
            candidate_count,
            view_count,
            3,
            self.patch_side,
            self.patch_side,
        )
        if (
            query_patches.shape != expected_query_shape
            or support_patches.shape != expected_support_shape
            or not torch.isfinite(query_patches).all()
            or not torch.isfinite(support_patches).all()
        ):
            raise ValueError("candidate RGB spatial patches do not match the fixed layout")
        edge_count = point_count * candidate_count * view_count
        flat_query_indices = active.query_image_indices[:, None, None].expand(
            -1, candidate_count, view_count
        ).reshape(-1)
        flat_query_xy = active.query_xy[:, None, None, :].expand(
            -1, candidate_count, view_count, -1
        ).reshape(-1, 2)
        flat_support_indices = active.support_image_indices.reshape(-1)
        flat_support_xy = active.support_xy.reshape(-1, 2)
        # Query RGB appearance is invariant across this point's fixed
        # candidate/support edges.  Encode it once, then gather the resulting
        # feature map per edge.  The old path re-ran the FPN for every edge,
        # which was pure redundant work and materially reduced two-GPU usage.
        query_texture_features = self.texture_encoder(query_patches)
        edge_to_point = torch.arange(point_count, device=self.device).repeat_interleave(
            candidate_count * view_count
        )
        flat_support_patches = support_patches.reshape(-1, 3, self.patch_side, self.patch_side)
        spatial_parts: list[torch.Tensor] = []
        raw_spatial_parts: list[torch.Tensor] = []
        spatial_residual_parts: list[torch.Tensor] = []
        dustbin_parts: list[torch.Tensor] = []
        context_parts: list[torch.Tensor] = []
        crop_valid_parts: list[torch.Tensor] = []
        for begin in range(0, edge_count, self.edge_chunk_size):
            end = min(begin + self.edge_chunk_size, edge_count)
            tensors = (
                flat_query_indices[begin:end],
                flat_query_xy[begin:end],
                flat_support_indices[begin:end],
                flat_support_xy[begin:end],
                query_texture_features[edge_to_point[begin:end]],
                flat_support_patches[begin:end],
            )
            if self.training and self.activation_checkpointing and torch.is_grad_enabled():
                raw_spatial, residual, non_dustbin, context, crop_valid = checkpoint(
                    self._edge_chunk_from_tensors, *tensors, use_reentrant=False
                )
            else:
                raw_spatial, residual, non_dustbin, context, crop_valid = self._edge_chunk(
                    query_image_indices=tensors[0],
                    query_xy=tensors[1],
                    support_image_indices=tensors[2],
                    support_xy=tensors[3],
                    query_texture_features=tensors[4],
                    support_rgb_patches=tensors[5],
                )
            spatial_parts.append(raw_spatial + residual)
            raw_spatial_parts.append(raw_spatial)
            spatial_residual_parts.append(residual)
            dustbin_parts.append(non_dustbin)
            context_parts.append(context)
            crop_valid_parts.append(crop_valid)
        spatial_logits = torch.cat(spatial_parts, dim=0).reshape(
            point_count, candidate_count, view_count, -1
        )
        raw_spatial_logits = torch.cat(raw_spatial_parts, dim=0).reshape_as(spatial_logits)
        spatial_residual_logits = torch.cat(spatial_residual_parts, dim=0).reshape_as(
            spatial_logits
        )
        non_dustbin_logits = torch.cat(dustbin_parts, dim=0).reshape(
            point_count, candidate_count, view_count
        )
        context_llr = torch.cat(context_parts, dim=0).reshape(
            point_count, candidate_count, view_count
        )
        crop_valid = torch.cat(crop_valid_parts, dim=0).reshape(
            point_count, candidate_count, view_count
        )
        full_joint = normalized_spatial_log_probabilities_with_dustbin(
            spatial_logits.reshape(-1, spatial_logits.shape[-1]),
            non_dustbin_logits.reshape(-1),
        ).reshape(point_count, candidate_count, view_count, -1)
        query_rgb_usable = self._rgb_window_usable(
            image_indices=active.query_image_indices,
            centers_xy=active.query_xy,
        )
        support_rgb_usable = self._rgb_window_usable(
            image_indices=active.support_image_indices.reshape(-1),
            centers_xy=active.support_xy.reshape(-1, 2),
        ).reshape(point_count, candidate_count, view_count)
        rgb_edge_usable = (
            active.support_view_valid
            & query_rgb_usable[:, None, None]
            & support_rgb_usable
        )
        context_edge_usable = crop_valid & active.support_view_valid
        joint = _source_safe_joint_log_probabilities(
            full_joint_log_probabilities=full_joint,
            raw_spatial_logits=raw_spatial_logits,
            rgb_edge_usable=rgb_edge_usable,
            context_edge_usable=context_edge_usable,
        )
        context_llr = torch.where(
            context_edge_usable,
            context_llr,
            torch.zeros_like(context_llr),
        )
        return CandidatePoseRGBSpatialEdgePrediction(
            spatial_logits=spatial_logits,
            non_dustbin_logits=non_dustbin_logits,
            joint_log_probabilities=joint,
            offsets_xy=self._offsets_xy,
            context_log_likelihood_ratios=context_llr,
            edge_usable=rgb_edge_usable | context_edge_usable,
            raw_spatial_logits=raw_spatial_logits,
            spatial_residual_logits=spatial_residual_logits,
            rgb_edge_usable=rgb_edge_usable,
            context_edge_usable=context_edge_usable,
        )
