"""Audit frozen CandidateMapletMatcher ordering semantics on real episodes.

The matcher contains several set-valued stages.  A checkpoint must therefore
be invariant to reordering non-anchor maplet nodes and support views, and
equivariant to reordering mutually exclusive top-L candidates.  This tool
uses only inference-time inputs and never reads poses, residuals, labels, or
score-selection targets.

It is an implementation-hygiene audit, not a localization experiment.  A
passing result only rules out an ordering shortcut in this legacy candidate
maplet branch; it does not promote the branch to the production pose selector.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.candidate_maplet_data import (
    CandidateMapletEpisodeStore,
)
from feature_extract.vfm.localization.candidate_maplet_matcher import (
    CandidateMapletBatch,
    CandidateMapletMatcher,
    CandidateMapletMatcherConfig,
)
from feature_extract.vfm.localization.candidate_maplet_schema import (
    candidate_maplet_inference_manifest_mismatches,
)


_SUPPORTED_CHECKPOINT_FORMATS = {
    "candidate_maplet_matcher_checkpoint_v5",
    "candidate_maplet_matcher_checkpoint_v6",
    "candidate_maplet_matcher_checkpoint_v7",
    "candidate_maplet_matcher_checkpoint_v8",
    "candidate_maplet_matcher_checkpoint_v9",
    "candidate_maplet_matcher_checkpoint_v10",
    "candidate_maplet_matcher_checkpoint_v11",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--detector_query_cache", required=True)
    parser.add_argument("--query_context_detector_cache", required=True)
    parser.add_argument("--support_feature_cache", required=True)
    parser.add_argument("--support_geometry_index", required=True)
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument("--maplet_support_index", required=True)
    parser.add_argument("--feature_artifact", required=True)
    parser.add_argument("--radio_intermediate_cache", default=None)
    parser.add_argument("--radio_final_context_cache", default=None)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample_group_count", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--rtol", type=float, default=2e-5)
    parser.add_argument("--anchor_min_changed_fraction", type=float, default=0.90)
    parser.add_argument("--support_view_count", type=int, default=2)
    parser.add_argument("--static_feature_count", type=int, default=74)
    parser.add_argument("--query_radius_px", type=float, default=96.0)
    parser.add_argument("--max_query_nodes", type=int, default=48)
    parser.add_argument("--max_support_tracks", type=int, default=33)
    parser.add_argument("--positive_threshold_px", type=float, default=2.0)
    parser.add_argument("--assignment_threshold_px", type=float, default=5.0)
    return parser.parse_args(argv)


def _inference_manifest(
    args: argparse.Namespace, store: CandidateMapletEpisodeStore
) -> dict[str, object]:
    return {
        "support_feature_cache_sha256": file_sha256_short(
            Path(args.support_feature_cache)
        ),
        "support_geometry_index_sha256": file_sha256_short(
            Path(args.support_geometry_index)
        ),
        "projected_landmark_bank_sha256": file_sha256_short(
            Path(args.projected_landmark_bank)
        ),
        "maplet_support_index_sha256": file_sha256_short(
            Path(args.maplet_support_index)
        ),
        "query_input_dim": int(store.query_input_dim),
        "support_input_dim": int(store.support_input_dim),
        "static_input_dim": int(store.static_input_dim),
        "candidate_top_k": int(store.candidate_top_k),
        "static_feature_names": list(store.static_feature_names),
        "positive_threshold_px": float(args.positive_threshold_px),
        "assignment_threshold_px": float(args.assignment_threshold_px),
        "support_view_count": int(args.support_view_count),
        "query_radius_px": float(args.query_radius_px),
        "max_query_nodes": int(args.max_query_nodes),
        "max_support_tracks": int(args.max_support_tracks),
    }


def _input_hashes(args: argparse.Namespace) -> dict[str, object]:
    paths: dict[str, Path | None] = {
        "proposals": Path(args.proposals),
        "detector_query_cache": Path(args.detector_query_cache),
        "query_context_detector_cache": Path(args.query_context_detector_cache),
        "support_feature_cache": Path(args.support_feature_cache),
        "support_geometry_index": Path(args.support_geometry_index),
        "projected_landmark_bank": Path(args.projected_landmark_bank),
        "maplet_support_index": Path(args.maplet_support_index),
        "feature_artifact": Path(args.feature_artifact),
        "radio_intermediate_cache": (
            None
            if args.radio_intermediate_cache is None
            else Path(args.radio_intermediate_cache)
        ),
        "radio_final_context_cache": (
            None
            if args.radio_final_context_cache is None
            else Path(args.radio_final_context_cache)
        ),
        "colmap_cameras": Path(args.colmap_model_dir) / "cameras.bin",
        "colmap_images": Path(args.colmap_model_dir) / "images.bin",
        "colmap_points3d": Path(args.colmap_model_dir) / "points3D.bin",
    }
    return {
        name: None if path is None else file_sha256_short(path)
        for name, path in paths.items()
    }


def _stratified_group_sample(
    group_query_ids: np.ndarray,
    *,
    sample_count: int,
    seed: int,
) -> np.ndarray:
    """Sample groups deterministically while covering every query when possible."""

    query_ids = np.asarray(group_query_ids).astype(str).reshape(-1)
    if int(sample_count) <= 0:
        raise ValueError("sample_group_count must be positive")
    if len(query_ids) == 0:
        raise ValueError("candidate-maplet store has no groups")
    requested = min(int(sample_count), len(query_ids))
    rng = np.random.default_rng(int(seed))
    unique = np.asarray(sorted(set(query_ids.tolist())))
    if requested < len(unique):
        selected_queries = unique[rng.choice(len(unique), size=requested, replace=False)]
    else:
        selected_queries = unique
    selected: list[int] = []
    for query_id in selected_queries.tolist():
        choices = np.flatnonzero(query_ids == str(query_id))
        selected.append(int(choices[rng.integers(len(choices))]))
    remaining = requested - len(selected)
    if remaining > 0:
        available = np.setdiff1d(
            np.arange(len(query_ids), dtype=np.int64),
            np.asarray(selected, dtype=np.int64),
            assume_unique=False,
        )
        selected.extend(
            rng.choice(available, size=remaining, replace=False).astype(np.int64).tolist()
        )
    return np.asarray(sorted(selected), dtype=np.int64)


def _non_anchor_orders(mask: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Return per-episode orders that leave node zero and padded suffixes fixed."""

    valid = mask.bool()
    if valid.ndim != 2 or not torch.all(valid[:, 0]):
        raise ValueError("node permutation requires a valid node-zero anchor")
    order = torch.arange(valid.shape[1], device=valid.device).expand_as(valid).clone()
    for row in range(int(valid.shape[0])):
        count = int(torch.sum(valid[row]).item())
        if not torch.all(valid[row, :count]) or torch.any(valid[row, count:]):
            raise ValueError("node masks must be a contiguous valid prefix")
        # With one non-anchor node no non-trivial permutation exists.
        if count <= 2:
            continue
        non_anchor = torch.arange(1, count, device=valid.device)
        shift = 1 + (int(seed) + row) % (count - 2)
        order[row, 1:count] = torch.roll(non_anchor, shifts=shift)
    return order


def _swap_anchor_orders(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Produce a deliberately invalid anchor-role perturbation for diagnostics."""

    valid = mask.bool()
    order = torch.arange(valid.shape[1], device=valid.device).expand_as(valid).clone()
    eligible = torch.sum(valid, dim=1) >= 2
    for row in torch.nonzero(eligible, as_tuple=False).reshape(-1).tolist():
        order[row, 0] = 1
        order[row, 1] = 0
    return order, eligible


def _take_nodes(values: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    if values.ndim == 2:
        return torch.gather(values, 1, order)
    if values.ndim == 3:
        return torch.gather(
            values,
            1,
            order.unsqueeze(2).expand(-1, -1, int(values.shape[2])),
        )
    raise ValueError("node tensors must have two or three dimensions")


def _reorder_inference_batch(
    batch: CandidateMapletBatch,
    *,
    query_order: torch.Tensor,
    support_order: torch.Tensor,
) -> CandidateMapletBatch:
    if (
        batch.target_track_indices is not None
        or batch.candidate_labels is not None
        or batch.anchor_residuals_px is not None
        or batch.candidate_visible is not None
    ):
        raise ValueError("permutation audit requires an inference-only episode batch")
    reordered = CandidateMapletBatch(
        query_features=_take_nodes(batch.query_features, query_order),
        query_mask=_take_nodes(batch.query_mask, query_order),
        support_features=_take_nodes(batch.support_features, support_order),
        support_mask=_take_nodes(batch.support_mask, support_order),
        static_features=batch.static_features,
        target_track_indices=None,
        candidate_labels=None,
        edge_indices=batch.edge_indices,
    )
    reordered.validate()
    return reordered


def _restore_pair_axis(
    values: torch.Tensor,
    *,
    query_inverse: torch.Tensor,
    support_inverse: torch.Tensor,
) -> torch.Tensor:
    restored = _take_nodes(values, query_inverse)
    return torch.gather(
        restored,
        2,
        support_inverse.unsqueeze(1).expand(
            -1, int(restored.shape[1]), -1
        ),
    )


def _restore_query_probabilities(
    values: torch.Tensor,
    *,
    query_inverse: torch.Tensor,
    support_inverse: torch.Tensor,
) -> torch.Tensor:
    support_count = int(support_inverse.shape[1])
    restored = _take_nodes(values, query_inverse)
    pair = torch.gather(
        restored[:, :, :support_count],
        2,
        support_inverse.unsqueeze(1).expand(-1, int(restored.shape[1]), -1),
    )
    return torch.cat([pair, restored[:, :, support_count:]], dim=2)


def _metric_state() -> dict[str, float | int]:
    return {
        "element_count": 0,
        "violation_count": 0,
        "absolute_sum": 0.0,
        "max_absolute_difference": 0.0,
        "max_relative_difference": 0.0,
    }


def _update_metric(
    state: dict[str, float | int],
    expected: torch.Tensor,
    actual: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> None:
    if expected.shape != actual.shape:
        raise ValueError(
            f"permutation output shape mismatch: expected {tuple(expected.shape)}, "
            f"actual {tuple(actual.shape)}"
        )
    difference = torch.abs(expected.float() - actual.float())
    tolerance = float(atol) + float(rtol) * torch.abs(expected.float())
    state["element_count"] = int(state["element_count"]) + int(difference.numel())
    state["violation_count"] = int(state["violation_count"]) + int(
        torch.sum(difference > tolerance).item()
    )
    state["absolute_sum"] = float(state["absolute_sum"]) + float(
        torch.sum(difference).item()
    )
    state["max_absolute_difference"] = max(
        float(state["max_absolute_difference"]), float(torch.max(difference).item())
    )
    relative = difference / torch.clamp(torch.abs(expected.float()), min=1e-8)
    state["max_relative_difference"] = max(
        float(state["max_relative_difference"]), float(torch.max(relative).item())
    )


def _finalize_metrics(
    states: Mapping[str, dict[str, float | int]]
) -> dict[str, object]:
    metrics: dict[str, object] = {}
    for name, state in states.items():
        count = int(state["element_count"])
        violations = int(state["violation_count"])
        metrics[name] = {
            "element_count": count,
            "violation_count": violations,
            "max_absolute_difference": float(state["max_absolute_difference"]),
            "mean_absolute_difference": float(state["absolute_sum"]) / max(count, 1),
            "max_relative_difference": float(state["max_relative_difference"]),
            "passed": bool(violations == 0),
        }
    return metrics


def _all_passed(metrics: Mapping[str, object]) -> bool:
    return all(bool(dict(value).get("passed", False)) for value in metrics.values())


def _group_candidate_orders(
    group_count: int, candidate_count: int, *, seed: int, device: torch.device
) -> torch.Tensor:
    if int(candidate_count) <= 0:
        raise ValueError("candidate count must be positive")
    orders = torch.arange(candidate_count, device=device).repeat(group_count, 1)
    if int(candidate_count) <= 1:
        return orders
    for row in range(group_count):
        shift = 1 + (int(seed) + row) % (candidate_count - 1)
        orders[row] = torch.roll(orders[row], shifts=shift)
    return orders


def _per_candidate_view_orders(
    candidate_count: int, view_count: int, *, seed: int, device: torch.device
) -> torch.Tensor:
    orders = torch.arange(view_count, device=device).repeat(candidate_count, 1)
    if int(view_count) <= 1:
        return orders
    for row in range(candidate_count):
        shift = 1 + (int(seed) + row) % (view_count - 1)
        orders[row] = torch.roll(orders[row], shifts=shift)
    return orders


def _take_candidate_axis(values: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    if values.ndim == 2:
        return torch.gather(values, 1, order)
    if values.ndim == 3:
        return torch.gather(
            values,
            1,
            order.unsqueeze(2).expand(-1, -1, int(values.shape[2])),
        )
    if values.ndim == 4:
        return torch.gather(
            values,
            1,
            order.unsqueeze(2)
            .unsqueeze(3)
            .expand(-1, -1, int(values.shape[2]), int(values.shape[3])),
        )
    raise ValueError("candidate tensors must have two to four dimensions")


def _take_view_axis(values: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    if values.ndim == 3:
        return torch.gather(values, 2, order)
    if values.ndim == 4:
        return torch.gather(
            values,
            2,
            order.unsqueeze(3).expand(-1, -1, -1, int(values.shape[3])),
        )
    raise ValueError("view tensors must have three or four dimensions")


def _resolve_candidate_groups(
    model: CandidateMapletMatcher,
    candidate_views: torch.Tensor,
    candidate_mask: torch.Tensor,
    candidate_prior_scores: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    if candidate_views.ndim != 4:
        raise ValueError("candidate views must have shape (G, L, V, C)")
    groups, candidates, views, channels = candidate_views.shape
    flat_views = candidate_views.reshape(groups * candidates, views, channels)
    _aggregated, view_weights = model.aggregate_candidate_views(flat_views)
    if bool(model.config.candidate_view_marginalization_enabled):
        resolved = model.resolve_candidate_view_sets(
            candidate_views,
            view_weights.reshape(groups, candidates, views),
            candidate_mask=candidate_mask,
            candidate_prior_scores=candidate_prior_scores,
        )
    else:
        resolved = model.resolve_candidate_sets(
            _aggregated.reshape(groups, candidates, channels),
            candidate_mask=candidate_mask,
            candidate_prior_scores=candidate_prior_scores,
        )
    return {name: value for name, value in resolved.items() if isinstance(value, torch.Tensor)}, view_weights.reshape(groups, candidates, views)


def _group_metric_fields(
    output: Mapping[str, torch.Tensor]
) -> tuple[str, ...]:
    fields = [
        "candidate_logits",
        "candidate_evidence_logits",
        "candidate_embeddings",
        "dustbin_logits",
    ]
    optional = (
        "top_l_availability_logits",
        "support_view_probabilities",
        "support_view_prior_probabilities",
        "identity_conditioned_support_view_probabilities",
    )
    fields.extend(name for name in optional if name in output)
    return tuple(fields)


def _restore_group_output(
    name: str,
    values: torch.Tensor,
    *,
    candidate_inverse: torch.Tensor | None,
    view_inverse: torch.Tensor | None,
) -> torch.Tensor:
    restored = values
    view_indexed_fields = {
        "support_view_probabilities",
        "support_view_prior_probabilities",
        "identity_conditioned_support_view_probabilities",
    }
    if view_inverse is not None and name in view_indexed_fields:
        restored = _take_view_axis(restored, view_inverse)
    if candidate_inverse is not None and restored.ndim >= 2 and int(restored.shape[1]) == int(
        candidate_inverse.shape[1]
    ):
        restored = _take_candidate_axis(restored, candidate_inverse)
    return restored


@torch.no_grad()
def _audit_checkpoint(
    model: CandidateMapletMatcher,
    store: CandidateMapletEpisodeStore,
    *,
    group_indices: np.ndarray,
    device: torch.device,
    batch_size: int,
    seed: int,
    atol: float,
    rtol: float,
    anchor_min_changed_fraction: float,
) -> dict[str, object]:
    model.eval()
    top_l = int(store.candidate_top_k)
    groups = np.asarray(group_indices, dtype=np.int64).reshape(-1)
    if len(groups) == 0:
        raise ValueError("permutation audit requires at least one candidate group")
    edge_matrix = groups[:, None] * top_l + np.arange(top_l, dtype=np.int64)[None]
    flat_edges = edge_matrix.reshape(-1)
    view_count = int(store.support_view_count)
    model_dim = int(model.config.model_dim)
    view_embeddings = np.empty((len(flat_edges), view_count, model_dim), dtype=np.float32)
    node_states = {
        name: _metric_state()
        for name in (
            "candidate_logits",
            "candidate_embeddings",
            "set_identity_candidate_embeddings",
            "pair_logits",
            "query_log_probabilities_batched",
        )
    }
    anchor_eligible_count = 0
    anchor_candidate_changed_count = 0
    anchor_assignment_changed_count = 0
    anchor_candidate_max_difference = 0.0
    anchor_assignment_max_difference = 0.0
    query_permutable_count = 0
    support_permutable_count = 0
    episode_count = 0

    for view_rank in range(view_count):
        for start in range(0, len(flat_edges), int(batch_size)):
            edges = flat_edges[start : start + int(batch_size)]
            ranks = np.full(edges.shape, view_rank, dtype=np.int64)
            original_batch = store.batch(edges, view_ranks=ranks).to(device)
            query_order = _non_anchor_orders(
                original_batch.query_mask, seed=int(seed) + 17 * view_rank
            )
            support_order = _non_anchor_orders(
                original_batch.support_mask, seed=int(seed) + 31 * view_rank
            )
            query_inverse = torch.argsort(query_order, dim=1)
            support_inverse = torch.argsort(support_order, dim=1)
            permuted_batch = _reorder_inference_batch(
                original_batch,
                query_order=query_order,
                support_order=support_order,
            )
            original = model(original_batch, return_ragged_query_probabilities=False)
            permuted = model(permuted_batch, return_ragged_query_probabilities=False)
            for name in (
                "candidate_logits",
                "candidate_embeddings",
                "set_identity_candidate_embeddings",
            ):
                value = original[name]
                changed = permuted[name]
                if not isinstance(value, torch.Tensor) or not isinstance(changed, torch.Tensor):
                    raise TypeError(f"matcher output is missing {name}")
                _update_metric(node_states[name], value, changed, atol=atol, rtol=rtol)
            original_pair = original["pair_logits"]
            permuted_pair = permuted["pair_logits"]
            original_probability = original["query_log_probabilities_batched"]
            permuted_probability = permuted["query_log_probabilities_batched"]
            if not all(
                isinstance(value, torch.Tensor)
                for value in (
                    original_pair,
                    permuted_pair,
                    original_probability,
                    permuted_probability,
                )
            ):
                raise TypeError("matcher output is missing pair or assignment tensors")
            _update_metric(
                node_states["pair_logits"],
                original_pair,
                _restore_pair_axis(
                    permuted_pair,
                    query_inverse=query_inverse,
                    support_inverse=support_inverse,
                ),
                atol=atol,
                rtol=rtol,
            )
            _update_metric(
                node_states["query_log_probabilities_batched"],
                original_probability,
                _restore_query_probabilities(
                    permuted_probability,
                    query_inverse=query_inverse,
                    support_inverse=support_inverse,
                ),
                atol=atol,
                rtol=rtol,
            )
            embeddings = original["set_identity_candidate_embeddings"]
            if not isinstance(embeddings, torch.Tensor):
                raise TypeError("matcher output is missing candidate embeddings")
            view_embeddings[start : start + len(edges), view_rank] = (
                embeddings.float().cpu().numpy()
            )

            query_permutable_count += int(
                torch.sum(torch.sum(original_batch.query_mask.bool(), dim=1) >= 3).item()
            )
            support_permutable_count += int(
                torch.sum(torch.sum(original_batch.support_mask.bool(), dim=1) >= 3).item()
            )
            episode_count += int(len(edges))

            query_swap, query_eligible = _swap_anchor_orders(original_batch.query_mask)
            support_swap, support_eligible = _swap_anchor_orders(original_batch.support_mask)
            swap_eligible = query_eligible & support_eligible
            if bool(torch.any(swap_eligible)):
                swapped_batch = _reorder_inference_batch(
                    original_batch,
                    query_order=query_swap,
                    support_order=support_swap,
                )
                swapped = model(swapped_batch, return_ragged_query_probabilities=False)
                original_candidate = original["candidate_logits"]
                swapped_candidate = swapped["candidate_logits"]
                original_assignment = original["query_log_probabilities_batched"]
                swapped_assignment = swapped["query_log_probabilities_batched"]
                if not all(
                    isinstance(value, torch.Tensor)
                    for value in (
                        original_candidate,
                        swapped_candidate,
                        original_assignment,
                        swapped_assignment,
                    )
                ):
                    raise TypeError("matcher output is missing anchor diagnostics")
                candidate_difference = torch.abs(
                    original_candidate[swap_eligible].float()
                    - swapped_candidate[swap_eligible].float()
                )
                assignment_difference = torch.abs(
                    original_assignment[swap_eligible, 0, 0].float()
                    - swapped_assignment[swap_eligible, 0, 0].float()
                )
                anchor_eligible_count += int(candidate_difference.numel())
                anchor_candidate_changed_count += int(
                    torch.sum(candidate_difference > float(atol)).item()
                )
                anchor_assignment_changed_count += int(
                    torch.sum(assignment_difference > float(atol)).item()
                )
                anchor_candidate_max_difference = max(
                    anchor_candidate_max_difference,
                    float(torch.max(candidate_difference).item()),
                )
                anchor_assignment_max_difference = max(
                    anchor_assignment_max_difference,
                    float(torch.max(assignment_difference).item()),
                )

    node_metrics = _finalize_metrics(node_states)
    candidate_views = torch.from_numpy(view_embeddings).to(device).reshape(
        len(groups), top_l, view_count, model_dim
    )
    candidate_mask = torch.from_numpy(store.valid_edges[groups]).to(device)
    prior_scores = torch.from_numpy(
        np.asarray(
            store.static_features[groups, :, int(model.config.candidate_prior_index)],
            dtype=np.float32,
        )
    ).to(device)
    original_group, original_view_weights = _resolve_candidate_groups(
        model, candidate_views, candidate_mask, prior_scores
    )
    fields = _group_metric_fields(original_group)

    edge_view_orders = _per_candidate_view_orders(
        len(flat_edges), view_count, seed=int(seed) + 101, device=device
    ).reshape(len(groups), top_l, view_count)
    edge_view_inverse = torch.argsort(edge_view_orders, dim=2)
    view_permuted_views = _take_view_axis(candidate_views, edge_view_orders)
    view_permuted_group, view_permuted_weights = _resolve_candidate_groups(
        model, view_permuted_views, candidate_mask, prior_scores
    )
    view_states = {name: _metric_state() for name in (*fields, "aggregation_weights")}
    _update_metric(
        view_states["aggregation_weights"],
        original_view_weights,
        _take_view_axis(view_permuted_weights, edge_view_inverse),
        atol=atol,
        rtol=rtol,
    )
    for name in fields:
        _update_metric(
            view_states[name],
            original_group[name],
            _restore_group_output(
                name,
                view_permuted_group[name],
                candidate_inverse=None,
                view_inverse=edge_view_inverse,
            ),
            atol=atol,
            rtol=rtol,
        )

    candidate_orders = _group_candidate_orders(
        len(groups), top_l, seed=int(seed) + 211, device=device
    )
    candidate_inverse = torch.argsort(candidate_orders, dim=1)
    rank_permuted_views = _take_candidate_axis(candidate_views, candidate_orders)
    rank_permuted_mask = _take_candidate_axis(candidate_mask, candidate_orders)
    rank_permuted_prior = _take_candidate_axis(prior_scores, candidate_orders)
    rank_permuted_group, _rank_weights = _resolve_candidate_groups(
        model,
        rank_permuted_views,
        rank_permuted_mask,
        rank_permuted_prior,
    )
    rank_states = {name: _metric_state() for name in fields}
    for name in fields:
        _update_metric(
            rank_states[name],
            original_group[name],
            _restore_group_output(
                name,
                rank_permuted_group[name],
                candidate_inverse=candidate_inverse,
                view_inverse=None,
            ),
            atol=atol,
            rtol=rtol,
        )

    combined_view_orders = _take_candidate_axis(edge_view_orders, candidate_orders)
    combined_view_inverse = torch.argsort(combined_view_orders, dim=2)
    combined_views = _take_view_axis(rank_permuted_views, combined_view_orders)
    combined_group, _combined_weights = _resolve_candidate_groups(
        model,
        combined_views,
        rank_permuted_mask,
        rank_permuted_prior,
    )
    combined_states = {name: _metric_state() for name in fields}
    for name in fields:
        _update_metric(
            combined_states[name],
            original_group[name],
            _restore_group_output(
                name,
                combined_group[name],
                candidate_inverse=candidate_inverse,
                view_inverse=combined_view_inverse,
            ),
            atol=atol,
            rtol=rtol,
        )

    anchor_candidate_fraction = float(anchor_candidate_changed_count) / max(
        anchor_eligible_count, 1
    )
    anchor_assignment_fraction = float(anchor_assignment_changed_count) / max(
        anchor_eligible_count, 1
    )
    anchor_semantics_passed = bool(
        anchor_eligible_count > 0
        and anchor_candidate_fraction >= float(anchor_min_changed_fraction)
        and anchor_assignment_fraction >= float(anchor_min_changed_fraction)
    )
    node_passed = _all_passed(node_metrics)
    view_metrics = _finalize_metrics(view_states)
    rank_metrics = _finalize_metrics(rank_states)
    combined_metrics = _finalize_metrics(combined_states)
    return {
        "sample": {
            "group_count": int(len(groups)),
            "edge_count": int(len(flat_edges)),
            "support_view_count": view_count,
            "query_nonanchor_permutable_episode_count": int(query_permutable_count),
            "support_nonanchor_permutable_episode_count": int(support_permutable_count),
            "episode_count_across_views": int(episode_count),
        },
        "nonanchor_node_permutation": {
            "metrics": node_metrics,
            "passed": node_passed,
        },
        "support_view_permutation": {
            "metrics": view_metrics,
            "passed": _all_passed(view_metrics),
        },
        "candidate_rank_permutation": {
            "metrics": rank_metrics,
            "passed": _all_passed(rank_metrics),
        },
        "candidate_rank_and_support_view_permutation": {
            "metrics": combined_metrics,
            "passed": _all_passed(combined_metrics),
        },
        "anchor_role_sensitivity_DIAGNOSTIC_ONLY": {
            "explicit_anchor_role_embedding": bool(
                model.config.explicit_anchor_role_embedding
            ),
            "eligible_episode_count": int(anchor_eligible_count),
            "candidate_logit_changed_fraction": anchor_candidate_fraction,
            "anchor_assignment_log_probability_changed_fraction": anchor_assignment_fraction,
            "candidate_logit_max_absolute_difference": anchor_candidate_max_difference,
            "anchor_assignment_log_probability_max_absolute_difference": anchor_assignment_max_difference,
            "minimum_changed_fraction": float(anchor_min_changed_fraction),
            "passed": anchor_semantics_passed,
        },
        "passed": bool(
            node_passed
            and _all_passed(view_metrics)
            and _all_passed(rank_metrics)
            and _all_passed(combined_metrics)
            and anchor_semantics_passed
        ),
    }


def _build_store(args: argparse.Namespace) -> CandidateMapletEpisodeStore:
    return CandidateMapletEpisodeStore(
        proposals=Path(args.proposals),
        detector_query_cache=Path(args.detector_query_cache),
        query_context_detector_cache=Path(args.query_context_detector_cache),
        support_feature_cache=Path(args.support_feature_cache),
        support_geometry_index=Path(args.support_geometry_index),
        projected_landmark_bank=Path(args.projected_landmark_bank),
        maplet_support_index=Path(args.maplet_support_index),
        feature_artifact=Path(args.feature_artifact),
        colmap_model_dir=Path(args.colmap_model_dir),
        radio_intermediate_cache=(
            None
            if args.radio_intermediate_cache is None
            else Path(args.radio_intermediate_cache)
        ),
        radio_final_context_cache=(
            None
            if args.radio_final_context_cache is None
            else Path(args.radio_final_context_cache)
        ),
        query_radius_px=float(args.query_radius_px),
        max_query_nodes=int(args.max_query_nodes),
        max_support_tracks=int(args.max_support_tracks),
        positive_threshold_px=float(args.positive_threshold_px),
        assignment_threshold_px=float(args.assignment_threshold_px),
        support_view_count=int(args.support_view_count),
        static_feature_count=int(args.static_feature_count),
        load_supervision=False,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if int(args.batch_size) <= 0 or int(args.sample_group_count) <= 0:
        raise ValueError("sample_group_count and batch_size must be positive")
    if float(args.atol) < 0.0 or float(args.rtol) < 0.0:
        raise ValueError("atol and rtol must be non-negative")
    if not 0.0 < float(args.anchor_min_changed_fraction) <= 1.0:
        raise ValueError("anchor_min_changed_fraction must be in (0, 1]")
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    checkpoint_paths = tuple(
        Path(value.strip())
        for value in str(args.checkpoints).split(",")
        if value.strip()
    )
    if not checkpoint_paths:
        raise ValueError("at least one checkpoint is required")
    store = _build_store(args)
    if int(store.edge_count) % int(store.candidate_top_k) != 0:
        raise ValueError("candidate-maplet edge count is not divisible by top-L")
    group_query_ids = store.edge_query_ids.reshape(-1, store.candidate_top_k)[:, 0]
    groups = _stratified_group_sample(
        group_query_ids,
        sample_count=int(args.sample_group_count),
        seed=int(args.seed),
    )
    manifest = _inference_manifest(args, store)
    checkpoint_results: list[dict[str, object]] = []
    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        checkpoint_format = str(checkpoint.get("format", ""))
        if checkpoint_format not in _SUPPORTED_CHECKPOINT_FORMATS:
            raise ValueError(f"unsupported checkpoint: {checkpoint_path}")
        checkpoint_manifest = dict(checkpoint.get("data_manifest") or {})
        mismatches = candidate_maplet_inference_manifest_mismatches(
            checkpoint_manifest, manifest
        )
        if mismatches:
            raise ValueError(
                "checkpoint is incompatible with permutation-audit store: "
                f"{json.dumps(mismatches, sort_keys=True)}"
            )
        config = CandidateMapletMatcherConfig(**dict(checkpoint["model_config"]))
        if (
            int(config.query_input_dim) != int(store.query_input_dim)
            or int(config.support_input_dim) != int(store.support_input_dim)
            or int(config.static_input_dim) != int(store.static_input_dim)
        ):
            raise ValueError("checkpoint tensor dimensions differ from episode store")
        model = CandidateMapletMatcher(config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        audit = _audit_checkpoint(
            model,
            store,
            group_indices=groups,
            device=device,
            batch_size=int(args.batch_size),
            seed=int(args.seed),
            atol=float(args.atol),
            rtol=float(args.rtol),
            anchor_min_changed_fraction=float(args.anchor_min_changed_fraction),
        )
        checkpoint_results.append(
            {
                "path": str(checkpoint_path),
                "sha256": file_sha256_short(checkpoint_path),
                "format": checkpoint_format,
                "seed": int(checkpoint.get("seed", -1)),
                "epoch": int(checkpoint.get("epoch", -1)),
                "model_config": config.to_dict(),
                "audit": audit,
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = {
        "format": "candidate_maplet_permutation_audit_v1",
        "protocol": {
            "inference_only": True,
            "ground_truth_loaded": False,
            "pose_or_ground_truth_used_for_sampling": False,
            "image_retrieval": False,
            "submap": False,
            "render": False,
            "purpose": "implementation_hygiene_not_pose_selector_promotion",
        },
        "inputs": {
            "hashes": _input_hashes(args),
            "inference_manifest": manifest,
        },
        "sample": {
            "seed": int(args.seed),
            "requested_group_count": int(args.sample_group_count),
            "selected_group_count": int(len(groups)),
            "total_group_count": int(len(group_query_ids)),
            "selected_query_count": int(len(set(group_query_ids[groups].tolist()))),
            "total_query_count": int(len(set(group_query_ids.tolist()))),
        },
        "tolerance": {
            "atol": float(args.atol),
            "rtol": float(args.rtol),
            "anchor_min_changed_fraction": float(args.anchor_min_changed_fraction),
        },
        "checkpoints": checkpoint_results,
        "passed": bool(all(bool(row["audit"]["passed"]) for row in checkpoint_results)),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
