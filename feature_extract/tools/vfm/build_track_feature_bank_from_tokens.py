"""Build a selected-track feature bank by sampling dense token maps."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.map_lifting import (
    TrackObservation,
    aggregate_selected_tracks,
    mapability_summary,
    save_selected_track_bank_npz,
)
from feature_extract.vfm.selector import LocalizableFeatureSelector
from feature_extract.vfm.tokens import TokenBankManifest
from feature_extract.vfm.track_feature_sampling import (
    load_colmap_track_observations_jsonl,
    sample_token_track_observations,
)


def _transform_sampled_observations(
    sampled: list[TrackObservation],
    transform: str,
    output_dim: int,
    seed: int,
) -> list[TrackObservation]:
    if transform == "identity":
        return sampled
    if not sampled:
        return sampled
    features = np.stack([obs.feature for obs in sampled], axis=0).astype(np.float32)
    channels = features.shape[1]
    if not 0 < output_dim <= channels:
        raise ValueError("output_dim must be in (0, feature_dim] for transformed banks")

    if transform == "first_channels":
        transformed = features[:, :output_dim]
    elif transform == "random_projection":
        rng = np.random.default_rng(seed)
        matrix = rng.normal(0.0, 1.0 / np.sqrt(output_dim), size=(channels, output_dim)).astype(
            np.float32
        )
        transformed = features @ matrix
    elif transform == "pca":
        centered = features - features.mean(axis=0, keepdims=True)
        _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
        transformed = centered @ vt[:output_dim].T
    else:
        raise ValueError(f"unsupported transform: {transform}")

    return [
        TrackObservation(
            track_id=obs.track_id,
            image_id=obs.image_id,
            feature=transformed[idx].astype(np.float32, copy=False),
            visible=obs.visible,
            geometry_valid=obs.geometry_valid,
            utility=obs.utility,
        )
        for idx, obs in enumerate(sampled)
    ]


def _infer_selector_dims(state_dict: dict[str, torch.Tensor]) -> tuple[int, int, int]:
    projection = state_dict.get("projection.weight")
    group_logits = state_dict.get("group_logits")
    if projection is None or group_logits is None:
        raise ValueError("selector checkpoint must contain projection.weight and group_logits")
    output_dim = int(projection.shape[0])
    input_dim = int(projection.shape[1])
    group_count = int(group_logits.numel())
    if group_count <= 0 or input_dim % group_count != 0:
        raise ValueError("cannot infer selector group_size from checkpoint")
    return input_dim, output_dim, input_dim // group_count


def _selector_transform_sampled_observations(
    sampled: list[TrackObservation],
    checkpoint_path: Path,
    device: str,
    batch_size: int,
) -> tuple[list[TrackObservation], dict[str, int | str]]:
    if not sampled:
        return sampled, {}
    if batch_size <= 0:
        raise ValueError("selector_batch_size must be positive")
    state_dict = torch.load(Path(checkpoint_path), map_location="cpu")
    input_dim, output_dim, group_size = _infer_selector_dims(state_dict)
    selector = LocalizableFeatureSelector(
        input_dim=input_dim,
        output_dim=output_dim,
        group_size=group_size,
    ).to(device)
    selector.load_state_dict(state_dict)
    selector.eval()

    features = np.stack([obs.feature for obs in sampled], axis=0).astype(np.float32)
    if features.shape[1] != input_dim:
        raise ValueError(f"selector expects {input_dim} input channels, got {features.shape[1]}")
    selected_chunks = []
    utility_chunks = []
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            batch = torch.from_numpy(features[start : start + batch_size]).to(device)
            batch = batch.view(batch.shape[0], batch.shape[1], 1, 1)
            output = selector(batch)
            selected_chunks.append(output.selected.flatten(1).cpu().numpy().astype(np.float32))
            utility_chunks.append(output.utility.flatten(1).mean(dim=1).cpu().numpy().astype(np.float32))
    transformed = np.concatenate(selected_chunks, axis=0)
    utilities = np.concatenate(utility_chunks, axis=0)
    transformed_observations = [
        TrackObservation(
            track_id=obs.track_id,
            image_id=obs.image_id,
            feature=transformed[idx],
            visible=obs.visible,
            geometry_valid=obs.geometry_valid,
            utility=float(utilities[idx]),
        )
        for idx, obs in enumerate(sampled)
    ]
    metadata = {
        "selector_input_dim": input_dim,
        "selector_output_dim": output_dim,
        "selector_group_size": group_size,
        "selector_checkpoint": str(checkpoint_path),
    }
    return transformed_observations, metadata


def _l2_normalize_sampled_observations(sampled: list[TrackObservation]) -> list[TrackObservation]:
    if not sampled:
        return sampled
    normalized = []
    for obs in sampled:
        feature = np.asarray(obs.feature, dtype=np.float32)
        norm = max(float(np.linalg.norm(feature)), 1e-6)
        normalized.append(
            TrackObservation(
                track_id=obs.track_id,
                image_id=obs.image_id,
                feature=feature / norm,
                visible=obs.visible,
                geometry_valid=obs.geometry_valid,
                utility=obs.utility,
            )
        )
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a sampled VFM track feature bank")
    parser.add_argument("--track_observations", required=True)
    parser.add_argument("--token_manifest", required=True)
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--min_observations", type=int, default=2)
    parser.add_argument("--missing", default="skip", choices=["skip", "error"])
    parser.add_argument(
        "--transform",
        default="identity",
        choices=["identity", "first_channels", "random_projection", "pca", "selector"],
    )
    parser.add_argument("--output_dim", type=int, default=0)
    parser.add_argument("--transform_seed", type=int, default=0)
    parser.add_argument("--l2_normalize", action="store_true")
    parser.add_argument("--selector_checkpoint", default=None)
    parser.add_argument("--selector_device", default="cpu")
    parser.add_argument("--selector_batch_size", type=int, default=1024)
    parser.add_argument("--output_bank", required=True)
    parser.add_argument("--summary_json", required=True)
    args = parser.parse_args()

    colmap_observations = load_colmap_track_observations_jsonl(Path(args.track_observations))
    sampled = sample_token_track_observations(
        colmap_observations,
        TokenBankManifest.from_json(Path(args.token_manifest)),
        layer_name=args.layer_name,
        missing=args.missing,
    )
    selector_metadata = {}
    if args.transform == "selector":
        if args.selector_checkpoint is None:
            raise ValueError("--selector_checkpoint is required when --transform selector")
        sampled, selector_metadata = _selector_transform_sampled_observations(
            sampled,
            checkpoint_path=Path(args.selector_checkpoint),
            device=args.selector_device,
            batch_size=args.selector_batch_size,
        )
    else:
        sampled = _transform_sampled_observations(
            sampled,
            transform=args.transform,
            output_dim=args.output_dim,
            seed=args.transform_seed,
        )
    if args.l2_normalize:
        sampled = _l2_normalize_sampled_observations(sampled)
    bank = aggregate_selected_tracks(sampled, min_observations=args.min_observations)
    save_selected_track_bank_npz(bank, Path(args.output_bank))

    summary = asdict(mapability_summary(bank))
    summary["input_observation_count"] = len(colmap_observations)
    summary["sampled_observation_count"] = len(sampled)
    summary["transform"] = args.transform
    summary["output_dim"] = summary["feature_dim"] if args.transform in {"identity", "selector"} else args.output_dim
    summary["transform_seed"] = args.transform_seed
    summary["l2_normalize"] = bool(args.l2_normalize)
    summary["input_files"] = {
        "track_observations": {
            "path": str(args.track_observations),
            "sha256": file_sha256_short(Path(args.track_observations)),
        },
        "token_manifest": {
            "path": str(args.token_manifest),
            "sha256": file_sha256_short(Path(args.token_manifest)),
        },
    }
    if args.selector_checkpoint:
        summary["input_files"]["selector_checkpoint"] = {
            "path": str(args.selector_checkpoint),
            "sha256": file_sha256_short(Path(args.selector_checkpoint)),
        }
    summary.update(selector_metadata)
    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
