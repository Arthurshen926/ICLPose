"""Run frozen candidate-maplet checkpoints on a new real-image query set.

This tool performs inference only. It does not inspect pose labels, select a
score strategy, tune a threshold, or run PnP. Query-specific artifact hashes
may differ from training, while support, landmark, feature-schema, and RADIO
projection identities are validated strictly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from feature_extract.tools.vfm.train_candidate_maplet_matcher import (
    _candidate_data_manifest,
    _compact_values,
    _predict_edges,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.candidate_maplet_data import (
    CandidateMapletEpisodeStore,
)
from feature_extract.vfm.localization.candidate_maplet_matcher import (
    CandidateMapletMatcher,
    CandidateMapletMatcherConfig,
)
from feature_extract.vfm.localization.candidate_maplet_schema import (
    candidate_maplet_inference_manifest_mismatches,
)
from feature_extract.vfm.localization.local_assignment_linear import (
    resolve_rescue_policy_scores,
)


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
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--baseline_strategy", default="alike_support_top2_mean")
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--support_view_count", type=int, default=2)
    parser.add_argument("--static_feature_count", type=int, default=74)
    parser.add_argument("--query_radius_px", type=float, default=96.0)
    parser.add_argument("--max_query_nodes", type=int, default=48)
    parser.add_argument("--max_support_tracks", type=int, default=33)
    parser.add_argument("--positive_threshold_px", type=float, default=2.0)
    parser.add_argument("--assignment_threshold_px", type=float, default=5.0)
    parser.add_argument("--rescue_action_margin_threshold", type=float, default=0.0)
    parser.add_argument("--no_amp", action="store_true")
    return parser.parse_args(argv)


def _radio_metadata(path: Path) -> dict[str, object]:
    with np.load(Path(path), allow_pickle=False) as data:
        return json.loads(str(data["metadata_json"].item()))


def _validate_radio_projection_lineage(
    checkpoint_manifest: dict[str, object],
    inference_radio_cache: Path | None,
) -> dict[str, object]:
    checkpoint_hash = checkpoint_manifest.get("radio_intermediate_cache_sha256")
    if checkpoint_hash is None:
        if inference_radio_cache is not None:
            raise ValueError("checkpoint did not use RADIO intermediate context")
        return {"mode": "not_used", "checkpoint_cache_sha256": None}
    if inference_radio_cache is None:
        raise ValueError("checkpoint requires a RADIO intermediate context cache")
    inference_hash = file_sha256_short(inference_radio_cache)
    if str(inference_hash) == str(checkpoint_hash):
        return {
            "mode": "same_cache",
            "checkpoint_cache_sha256": str(checkpoint_hash),
            "inference_cache_sha256": str(inference_hash),
        }
    metadata = _radio_metadata(inference_radio_cache)
    if str(metadata.get("projection_source_cache_sha256", "")) != str(
        checkpoint_hash
    ):
        raise ValueError(
            "external RADIO cache must reuse the checkpoint training cache as "
            "projection_source_cache"
        )
    if str(metadata.get("support_descriptor_source", "")) != (
        "reused_projection_source_cache"
    ):
        raise ValueError("external RADIO cache did not reuse support descriptors")
    return {
        "mode": "query_only_cache_with_training_projection_lineage",
        "checkpoint_cache_sha256": str(checkpoint_hash),
        "inference_cache_sha256": str(inference_hash),
        "projection_source_cache_sha256": str(
            metadata.get("projection_source_cache_sha256")
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    checkpoint_paths = tuple(
        Path(value.strip())
        for value in str(args.checkpoints).split(",")
        if value.strip()
    )
    devices = tuple(
        value.strip() for value in str(args.devices).split(",") if value.strip()
    )
    if not checkpoint_paths or not devices:
        raise ValueError("at least one checkpoint and one device are required")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    store = CandidateMapletEpisodeStore(
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
        query_radius_px=float(args.query_radius_px),
        max_query_nodes=int(args.max_query_nodes),
        max_support_tracks=int(args.max_support_tracks),
        positive_threshold_px=float(args.positive_threshold_px),
        assignment_threshold_px=float(args.assignment_threshold_px),
        support_view_count=int(args.support_view_count),
        static_feature_count=int(args.static_feature_count),
        load_supervision=False,
    )
    inference_manifest = _candidate_data_manifest(args, store)
    inference_edges = np.arange(store.edge_count, dtype=np.int64)
    predictions_by_checkpoint: list[dict[str, np.ndarray]] = []
    checkpoint_rows: list[dict[str, object]] = []
    checkpoint_formats: set[str] = set()
    model_configs: list[dict[str, object]] = []
    radio_lineages: list[dict[str, object]] = []
    for index, checkpoint_path in enumerate(checkpoint_paths):
        device = torch.device(devices[index % len(devices)])
        checkpoint = torch.load(checkpoint_path, map_location=device)
        checkpoint_format = str(checkpoint.get("format", ""))
        if checkpoint_format not in {
            "candidate_maplet_matcher_checkpoint_v5",
            "candidate_maplet_matcher_checkpoint_v6",
            "candidate_maplet_matcher_checkpoint_v7",
        }:
            raise ValueError(f"unsupported checkpoint: {checkpoint_path}")
        checkpoint_manifest = dict(checkpoint.get("data_manifest") or {})
        mismatches = candidate_maplet_inference_manifest_mismatches(
            checkpoint_manifest, inference_manifest
        )
        if mismatches:
            raise ValueError(
                "checkpoint is incompatible with external query inference: "
                f"{json.dumps(mismatches, sort_keys=True)}"
            )
        radio_lineage = _validate_radio_projection_lineage(
            checkpoint_manifest,
            None
            if args.radio_intermediate_cache is None
            else Path(args.radio_intermediate_cache),
        )
        config_payload = dict(checkpoint["model_config"])
        config = CandidateMapletMatcherConfig(**config_payload)
        if (
            int(config.query_input_dim) != int(store.query_input_dim)
            or int(config.support_input_dim) != int(store.support_input_dim)
            or int(config.static_input_dim) != int(store.static_input_dim)
        ):
            raise ValueError("checkpoint tensor dimensions differ from inference data")
        model = CandidateMapletMatcher(config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        predictions_by_checkpoint.append(
            _predict_edges(
                model,
                store,
                inference_edges,
                device=device,
                batch_size=int(args.eval_batch_size),
                use_amp=bool(device.type == "cuda" and not args.no_amp),
            )
        )
        checkpoint_formats.add(checkpoint_format)
        model_configs.append(config_payload)
        radio_lineages.append(radio_lineage)
        checkpoint_rows.append(
            {
                "path": str(checkpoint_path),
                "sha256": file_sha256_short(checkpoint_path),
                "format": checkpoint_format,
                "seed": int(checkpoint.get("seed", -1)),
                "epoch": int(checkpoint.get("epoch", -1)),
                "checkpoint_role": checkpoint.get("checkpoint_role"),
                "training_selection": checkpoint.get("selection"),
                "training_data_manifest": checkpoint_manifest,
                "radio_projection_lineage": radio_lineage,
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if len(checkpoint_formats) != 1 or any(
        payload != model_configs[0] for payload in model_configs[1:]
    ):
        raise ValueError("ensemble checkpoints use different model contracts")

    ensemble: dict[str, np.ndarray] = {}
    member_scores: dict[str, np.ndarray] = {}
    for checkpoint_index, predictions in enumerate(predictions_by_checkpoint):
        for strategy, values in predictions.items():
            member_scores[f"member_{checkpoint_index}__{strategy}"] = np.asarray(
                values, dtype=np.float32
            ).reshape(len(store.selected_rows), store.candidate_top_k)
    for strategy in predictions_by_checkpoint[0]:
        stacked = np.stack(
            [values[strategy] for values in predictions_by_checkpoint], axis=0
        )
        finite = np.isfinite(stacked)
        count = np.sum(finite, axis=0)
        values = np.full(stacked.shape[1:], -np.inf, dtype=np.float32)
        accepted = count > 0
        values[accepted] = (
            np.sum(np.where(finite, stacked, 0.0), axis=0)[accepted]
            / count[accepted]
        ).astype(np.float32)
        ensemble[strategy] = values.reshape(
            len(store.selected_rows), store.candidate_top_k
        )
    baseline_scores = _compact_values(
        store.proposals[f"strategy__{args.baseline_strategy}"], store
    )
    if {
        "rescue_candidate_probability",
        "rescue_keep_probability_DIAGNOSTIC_ONLY",
    }.issubset(ensemble):
        selected, resolved, _switched, _margin, action_scores = (
            resolve_rescue_policy_scores(
                ensemble["rescue_candidate_probability"],
                ensemble["rescue_keep_probability_DIAGNOSTIC_ONLY"][:, 0],
                baseline_scores,
                action_margin_threshold=float(args.rescue_action_margin_threshold),
                valid_mask=store.valid_edges,
            )
        )
        ensemble["rescue_policy_resolved"] = resolved
        ensemble["rescue_action_probability"] = action_scores
    else:
        selected = np.argmax(
            np.where(store.valid_edges, baseline_scores, -np.inf), axis=1
        ).astype(np.int64)

    scores_path = output_dir / "inference_scores.npz"
    np.savez(
        scores_path,
        **member_scores,
        **{
            f"ensemble__{name}": np.asarray(values, dtype=np.float32)
            for name, values in ensemble.items()
        },
        baseline_scores=baseline_scores.astype(np.float32),
        selected_columns=np.asarray(selected, dtype=np.int64),
    )
    summary = {
        "stage": "candidate_maplet_external_query_inference",
        "protocol": {
            "inference_only": True,
            "baseline_strategy": str(args.baseline_strategy),
            "ground_truth_pose_read": False,
            "supervision_arrays_loaded": False,
            "score_strategy_selected": False,
            "member_scores_exported": True,
            "threshold_tuned": False,
            "pnp_run": False,
            "measurement": False,
            "render": False,
            "image_retrieval": False,
            "submap": False,
        },
        "data_manifest": inference_manifest,
        "query_set": {
            "query_count": int(len(np.unique(store.query_ids[store.selected_rows]))),
            "query_point_count": int(len(store.selected_rows)),
            "candidate_top_k": int(store.candidate_top_k),
            "edge_count": int(store.edge_count),
        },
        "checkpoints": checkpoint_rows,
        "member_score_prefixes": [
            f"member_{index}" for index in range(len(checkpoint_rows))
        ],
        "model_config": model_configs[0],
        "radio_projection_lineages": radio_lineages,
        "outputs": {
            "scores": str(scores_path),
            "scores_sha256": file_sha256_short(scores_path),
            "summary": str(output_dir / "summary.json"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
