"""Train a compact chart-local RADIO head from mapping-only metric-UV identities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from feature_extract.vfm.localization_goal_maplet.geometry_native_planar_map import GeometryNativePlanarMap
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.plane_visibility_atlas import PlaneVisibilityAtlas


_OBSERVATION_BANK_ARRAY_NAMES = (
    "observation_offsets", "token_ids", "world_points", "radio_features",
    "world_point_covariance_m2", "plane_pixel_purity",
    "plane_depth_dispersion_m",
)


def _load_observation_bank(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {*_OBSERVATION_BANK_ARRAY_NAMES, "metadata_json"}:
            raise ValueError("mapping observation bank schema differs")
        metadata = json.loads(str(data["metadata_json"].item()))
        arrays = {name: np.asarray(data[name]) for name in _OBSERVATION_BANK_ARRAY_NAMES}
    content = dict(metadata)
    claimed_content = content.pop("content_sha256", None)
    if (
        metadata.get("artifact_type") != "goal_maplet_plane_pnp_observation_bank_v2"
        or metadata.get("plane_specific_geometry") is not True
        or metadata.get("uses_query_pose_or_ground_truth") is not False
        or arrays_sha256(arrays) != metadata.get("arrays_sha256")
        or canonical_json_sha256(content) != claimed_content
    ):
        raise ValueError("mapping observation bank lineage differs")
    return arrays, metadata


def _pair_inventory(
    identity: np.ndarray,
    observation: np.ndarray,
    route: np.ndarray,
    selected_routes: set[str],
) -> dict[str, np.ndarray]:
    rows = np.flatnonzero(np.isin(route[observation], sorted(selected_routes)))
    order = rows[np.lexsort((observation[rows], identity[rows]))]
    pair_start = np.r_[True, (identity[order][1:] != identity[order][:-1]) | (observation[order][1:] != observation[order][:-1])]
    start = np.flatnonzero(pair_start)
    pair_identity = identity[order[start]]
    pair_observation = observation[order[start]]
    pair_end = np.r_[start[1:], len(order)]
    identity_start = np.flatnonzero(np.r_[True, pair_identity[1:] != pair_identity[:-1]])
    identity_end = np.r_[identity_start[1:], len(pair_identity)]
    keep = identity_end - identity_start >= 2
    return {
        "token_order": order,
        "pair_start": start,
        "pair_end": pair_end,
        "pair_identity": pair_identity,
        "pair_observation": pair_observation,
        "identity_start": identity_start[keep],
        "identity_end": identity_end[keep],
        "eligible_identity": pair_identity[identity_start[keep]],
    }


def _sample_triplets(
    inventory: dict[str, np.ndarray],
    identity_plane: np.ndarray,
    rng: np.random.Generator,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eligible = inventory["eligible_identity"]
    plane_groups: dict[int, np.ndarray] = {}
    for plane in np.unique(identity_plane[eligible]):
        plane_groups[int(plane)] = np.flatnonzero(identity_plane[eligible] == plane)
    hard_negative_eligible = np.concatenate([
        groups for groups in plane_groups.values() if len(groups) >= 2
    ]) if plane_groups else np.zeros(0, np.int64)
    if len(hard_negative_eligible) < batch_size:
        raise ValueError("not enough same-plane hard-negative identities for one training batch")
    group_rows = rng.choice(hard_negative_eligible, size=batch_size, replace=False)

    def token_from_pair(pair: int) -> int:
        lo = int(inventory["pair_start"][pair]); hi = int(inventory["pair_end"][pair])
        return int(inventory["token_order"][rng.integers(lo, hi)])

    anchors, positives, negatives = [], [], []
    for group in group_rows.tolist():
        lo = int(inventory["identity_start"][group]); hi = int(inventory["identity_end"][group])
        pair = rng.choice(np.arange(lo, hi), size=2, replace=False)
        anchors.append(token_from_pair(int(pair[0])))
        positives.append(token_from_pair(int(pair[1])))
        plane = int(identity_plane[int(eligible[group])])
        candidates = plane_groups[plane]
        candidates = candidates[candidates != group]
        if not len(candidates):
            raise AssertionError("sampled identity lacks a same-plane hard negative")
        negative_group = int(rng.choice(candidates))
        nlo = int(inventory["identity_start"][negative_group])
        nhi = int(inventory["identity_end"][negative_group])
        negatives.append(token_from_pair(int(rng.integers(nlo, nhi))))
    return np.asarray(anchors), np.asarray(positives), np.asarray(negatives)


def _cosine_summary(feature: np.ndarray, rows: tuple[np.ndarray, ...], weight: np.ndarray | None) -> dict[str, float]:
    values = []
    for selected in rows:
        x = np.asarray(feature[selected], np.float32)
        if weight is not None:
            x = x @ weight.T
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
        values.append(x)
    positive = np.sum(values[0] * values[1], axis=1)
    negative = np.sum(values[0] * values[2], axis=1)
    return {
        "positive_cosine_mean": float(np.mean(positive)),
        "same_plane_negative_cosine_mean": float(np.mean(negative)),
        "positive_over_negative_fraction": float(np.mean(positive > negative)),
        "mean_positive_negative_margin": float(np.mean(positive - negative)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation_bank", type=Path, required=True)
    parser.add_argument("--visibility_atlas", type=Path, required=True)
    parser.add_argument("--planar_map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation_route", default="seq9")
    parser.add_argument("--cell_size_m", type=float, default=0.5)
    parser.add_argument("--output_dimension", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch_size", type=int, default=384)
    parser.add_argument("--seed", type=int, default=260903)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite chart-local projection")
    if args.output_dimension < 2 or args.cell_size_m <= 0 or args.steps < 1 or args.batch_size < 2:
        raise ValueError("invalid chart-local projection configuration")

    visibility, visibility_meta = PlaneVisibilityAtlas.load_npz(args.visibility_atlas)
    planes = GeometryNativePlanarMap.load_npz(args.planar_map)
    bank, bank_meta = _load_observation_bank(args.observation_bank)
    offsets = np.asarray(bank["observation_offsets"], np.int64)
    points = np.asarray(bank["world_points"], np.float64)
    features = np.asarray(bank["radio_features"], np.float32)
    if (
        bank_meta.get("visibility_atlas_content_sha256") != visibility_meta.get("content_sha256")
        or bank_meta.get("visibility_atlas_file_sha256") != file_sha256(args.visibility_atlas)
        or visibility_meta.get("planar_map_file_sha256") != file_sha256(args.planar_map)
        or len(offsets) != len(visibility.view_names) + 1
        or len(points) != len(features) or features.ndim != 2
        or bank["token_ids"].shape != (len(points),)
        or bank["world_point_covariance_m2"].shape != (len(points), 3, 3)
        or bank["plane_pixel_purity"].shape != (len(points),)
        or bank["plane_depth_dispersion_m"].shape != (len(points),)
    ):
        raise ValueError("mapping observation bank differs")

    observation = np.repeat(np.arange(len(visibility.view_names)), np.diff(offsets))
    observation_plane = np.repeat(np.arange(len(planes.plane_ids)), np.diff(visibility.plane_offsets))
    plane = observation_plane[observation]
    uv = np.empty((len(points), 2), np.float64)
    for row in range(len(planes.plane_ids)):
        selected = np.flatnonzero(plane == row)
        uv[selected] = (points[selected] - planes.centers_world[row]) @ planes.frames_world[row, :2].T
    cell = np.floor(uv / float(args.cell_size_m)).astype(np.int64)
    identity_key = np.c_[plane, cell]
    unique_identity, identity = np.unique(identity_key, axis=0, return_inverse=True)
    identity_plane = unique_identity[:, 0]
    route_per_observation = np.asarray([
        str(name).split("__", 1)[0] for name in visibility.view_names.astype(str)
    ])
    all_routes = sorted(set(route_per_observation.tolist()))
    fit_routes = set(all_routes) - {str(args.validation_route)}
    if str(args.validation_route) not in all_routes or not fit_routes:
        raise ValueError("validation route does not form a mapping-only split")
    fit = _pair_inventory(identity, observation, route_per_observation, fit_routes)
    validation = _pair_inventory(identity, observation, route_per_observation, {str(args.validation_route)})
    fit_plane, fit_plane_count = np.unique(
        identity_plane[fit["eligible_identity"]], return_counts=True,
    )
    validation_plane, validation_plane_count = np.unique(
        identity_plane[validation["eligible_identity"]], return_counts=True,
    )
    fit_hard_negative_eligible = int(np.count_nonzero(np.isin(
        identity_plane[fit["eligible_identity"]], fit_plane[fit_plane_count >= 2],
    )))
    validation_hard_negative_eligible = int(np.count_nonzero(np.isin(
        identity_plane[validation["eligible_identity"]],
        validation_plane[validation_plane_count >= 2],
    )))
    rng = np.random.default_rng(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer = torch.nn.Linear(features.shape[1], int(args.output_dimension), bias=False, device=device)
    torch.nn.init.orthogonal_(layer.weight)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=3e-3, weight_decay=1e-4)
    losses = []
    for step in range(int(args.steps)):
        sampled = _sample_triplets(fit, identity_plane, rng, int(args.batch_size))
        value = [F.normalize(torch.from_numpy(features[row]).to(device), dim=1) for row in sampled]
        anchor, positive, negative = [F.normalize(layer(item), dim=1) for item in value]
        logits = anchor @ positive.T / 0.07
        target = torch.arange(len(anchor), device=device)
        contrastive = 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))
        triplet = F.softplus((torch.sum(anchor * negative, dim=1) - torch.sum(anchor * positive, dim=1)) / 0.07).mean()
        loss = contrastive + triplet
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if (step + 1) % 200 == 0:
            print(f"step {step + 1}/{args.steps} loss={np.mean(losses[-200:]):.6f}", flush=True)

    weight = layer.weight.detach().cpu().numpy().astype(np.float32)
    validation_rows = _sample_triplets(
        validation, identity_plane, np.random.default_rng(int(args.seed) + 1),
        min(8192, validation_hard_negative_eligible),
    )
    raw = _cosine_summary(features, validation_rows, None)
    projected = _cosine_summary(features, validation_rows, weight)
    arrays = {"weight": weight}
    metadata = {
        "artifact_type": "goal_maplet_chart_local_radio_projection_v1",
        "input_dimension": int(features.shape[1]),
        "output_dimension": int(weight.shape[0]),
        "training_objective": "symmetric_InfoNCE_plus_same_physical_plane_hard_negative_softplus",
        "positive_identity": "same_finite_plane_and_same_floor_metric_uv_cell_across_distinct_mapping_observations",
        "hard_negative_identity": "same_finite_plane_different_metric_uv_cell",
        "cell_size_m": float(args.cell_size_m),
        "fit_mapping_routes": sorted(fit_routes),
        "validation_mapping_route": str(args.validation_route),
        "fit_validation_route_disjoint": True,
        "query_pose_depth_or_ground_truth_read": False,
        "mapping_rgb_stored": False,
        "source_view_identity_retained_at_runtime": False,
        "steps": int(args.steps), "batch_size": int(args.batch_size), "seed": int(args.seed),
        "fit_eligible_identity_count": int(len(fit["identity_start"])),
        "validation_eligible_identity_count": int(len(validation["identity_start"])),
        "fit_same_plane_hard_negative_eligible_identity_count": fit_hard_negative_eligible,
        "validation_same_plane_hard_negative_eligible_identity_count": validation_hard_negative_eligible,
        "validation_raw_1280d": raw,
        "validation_learned_projection": projected,
        "observation_bank_file_sha256": file_sha256(args.observation_bank),
        "observation_bank_content_sha256": bank_meta.get("content_sha256"),
        "visibility_atlas_file_sha256": file_sha256(args.visibility_atlas),
        "visibility_atlas_content_sha256": visibility_meta.get("content_sha256"),
        "planar_map_file_sha256": file_sha256(args.planar_map),
        "arrays_sha256": arrays_sha256(arrays),
    }
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
