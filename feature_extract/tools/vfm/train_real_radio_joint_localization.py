"""Train real-image RADIO selector + coarse + measurement jointly from full joint caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.tools.vfm.build_real_radio_joint_cache import (
    REFERENCED_MANIFEST_FORMAT,
    RealRadioReferencedJointSampleProvider,
    SfMTrackObservationIndex,
    load_track_observation_index,
)
from feature_extract.vfm.matcha_coarse_fine_adapter import save_matcha_coarse_fine_adapter
from feature_extract.vfm.landmark_feature_aggregation import LandmarkAggregationConfig, TrackPrototypeBuilder
from feature_extract.vfm.localization.descriptor_space import token_feature_source_config
from feature_extract.vfm.matcha_joint_training import (
    MatchaJointTrainingConfig,
    MatchaJointTrainingSet,
    _materialize_index_only_joint_training_set,
    joint_run_as_coarse_fine_adapter_run,
    load_matcha_joint_model,
    load_matcha_joint_training_set_npz,
    merge_matcha_joint_training_sets,
    save_matcha_joint_model,
    train_matcha_joint_model,
    train_matcha_joint_model_from_manifest,
    train_matcha_joint_model_from_sample_provider,
)
from feature_extract.vfm.tokens import TokenBankManifest


def validate_frozen_landmark_bank_contract(
    bank_path: Path,
    *,
    warm_start_checkpoint: Path | None,
    support_observations: Path,
    expected_source_image_count: int | None = None,
) -> dict[str, object]:
    with np.load(Path(bank_path), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item())) if "metadata_json" in data else {}
        descriptor_dim = int(np.asarray(data["features"]).shape[1])
        landmark_count = int(np.asarray(data["track_ids"]).shape[0])
    manifest = metadata.get("descriptor_space_manifest")
    if not isinstance(manifest, dict) or int(manifest.get("version", -1)) < 2:
        raise ValueError("frozen landmark bank requires a descriptor-space v2 manifest")
    if manifest.get("projection_source") != "projected_observation_full_map":
        raise ValueError("frozen landmark bank must use full-map projected observations")
    expected_track_hash = file_sha256_short(Path(support_observations))
    if metadata.get("track_observations_sha256") != expected_track_hash:
        raise ValueError(
            "frozen landmark bank support-observation mismatch: "
            f"expected={expected_track_hash!r}, got={metadata.get('track_observations_sha256')!r}"
        )
    if warm_start_checkpoint is not None:
        expected_checkpoint_hash = file_sha256_short(Path(warm_start_checkpoint))
        if metadata.get("matcha_joint_checkpoint_sha256") != expected_checkpoint_hash:
            raise ValueError(
                "frozen landmark bank was not built from the warm-start checkpoint: "
                f"expected={expected_checkpoint_hash!r}, got={metadata.get('matcha_joint_checkpoint_sha256')!r}"
            )
    if expected_source_image_count is not None and int(metadata.get("source_image_count", -1)) != int(
        expected_source_image_count
    ):
        raise ValueError(
            "frozen landmark bank source-image count does not match the allowed training support split: "
            f"expected={int(expected_source_image_count)}, got={metadata.get('source_image_count')!r}"
        )
    return {
        "path": str(bank_path),
        "descriptor_space_id": str(metadata.get("descriptor_space_id", "")),
        "landmark_count": int(landmark_count),
        "descriptor_dimension": int(descriptor_dim),
        "support_observations": str(support_observations),
        "support_observations_sha256": expected_track_hash,
        "source_image_count": int(metadata.get("source_image_count", 0)),
        "warm_start_checkpoint_sha256": str(metadata.get("matcha_joint_checkpoint_sha256", "")),
        "validated": True,
    }


class RealRadioMultiViewEpisodeProvider:
    """Group one query with distinct real-image support views on demand."""

    def __init__(
        self,
        provider: RealRadioReferencedJointSampleProvider,
        *,
        support_pairs: int,
        min_support_pairs: int = 2,
        seed: int = 0,
        support_selection: str = "manifest",
        episodes_per_query: int = 4,
        sfm_candidate_pool_size: int = 32,
        retrieval_tracks_per_episode: int = 16,
        allowed_support_image_ids: set[str] | None = None,
    ) -> None:
        if not 2 <= int(support_pairs) <= 8:
            raise ValueError("landmark episode support_pairs must be in [2, 8]")
        if not 2 <= int(min_support_pairs) <= int(support_pairs):
            raise ValueError("landmark episode min_support_pairs must be in [2, support_pairs]")
        if str(support_selection) not in {"manifest", "sfm_track_overlap", "sfm_track_episode"}:
            raise ValueError(
                "landmark episode support_selection must be 'manifest', "
                "'sfm_track_overlap', or 'sfm_track_episode'"
            )
        if int(episodes_per_query) <= 0:
            raise ValueError("landmark episode episodes_per_query must be positive")
        if int(sfm_candidate_pool_size) < int(support_pairs):
            raise ValueError("sfm_candidate_pool_size must be at least support_pairs")
        if int(retrieval_tracks_per_episode) <= 0:
            raise ValueError("retrieval_tracks_per_episode must be positive")
        records = list(dict(getattr(provider, "metadata", {})).get("records", []))
        if not records:
            raise ValueError("referenced provider metadata must contain records for episodic grouping")
        grouped: dict[str, dict[str, int]] = {}
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                continue
            query_id = str(record.get("query_id", ""))
            reference_id = str(record.get("reference_image_id", ""))
            if not query_id or not reference_id:
                raise ValueError("every episodic record requires query_id and reference_image_id")
            if query_id == reference_id:
                raise ValueError(f"query image leaked into its support set: {query_id!r}")
            grouped.setdefault(query_id, {}).setdefault(reference_id, int(index))
        episodes: list[tuple[str, tuple[object, ...], int | None]] = []
        repeated_track_counts: list[int] = []
        target_track_support_counts: list[int] = []
        complete_track_counts: list[int] = []
        if str(support_selection) == "manifest":
            for query_id in sorted(grouped):
                indices = np.asarray(list(grouped[query_id].values()), dtype=np.int64)
                if indices.size < int(min_support_pairs):
                    continue
                digest = hashlib.sha256(f"{int(seed)}:{query_id}".encode("utf-8")).digest()
                rng = np.random.default_rng(int.from_bytes(digest[:8], byteorder="little", signed=False))
                indices = indices[rng.permutation(indices.size)]
                for start in range(0, int(indices.size), int(support_pairs)):
                    selected = indices[start : start + int(support_pairs)].tolist()
                    if len(selected) < int(min_support_pairs):
                        for candidate in indices.tolist():
                            if int(candidate) not in selected:
                                selected.append(int(candidate))
                            if len(selected) >= int(min_support_pairs):
                                break
                    episodes.append((query_id, tuple(int(value) for value in selected), None))
        else:
            observation_index = getattr(provider, "track_observation_index", None)
            if observation_index is None:
                raise ValueError("sfm_track_overlap support selection requires a full SfM observation index")
            images_by_track: dict[int, list[str]] = {}
            tracks_by_image: dict[str, set[int]] = {}
            effective_allowed_support_ids = (
                set(grouped)
                if allowed_support_image_ids is None
                else {str(image_id) for image_id in allowed_support_image_ids}
            )
            for image_id, image_observations in observation_index.by_image.items():
                track_set = {int(value) for value in image_observations.track_ids.tolist()}
                tracks_by_image[str(image_id)] = track_set
                for track_id in track_set:
                    images_by_track.setdefault(int(track_id), []).append(str(image_id))
            for query_id in sorted(grouped):
                query_tracks = tracks_by_image.get(str(query_id), set())
                if str(support_selection) == "sfm_track_episode":
                    eligible_tracks: list[tuple[int, list[str]]] = []
                    for track_id in query_tracks:
                        support_images = sorted(
                            {
                                str(image_id)
                                for image_id in images_by_track.get(int(track_id), [])
                                if str(image_id) != str(query_id)
                                and str(image_id) in effective_allowed_support_ids
                            }
                        )
                        if len(support_images) >= int(min_support_pairs):
                            eligible_tracks.append((int(track_id), support_images))
                    eligible_tracks.sort(
                        key=lambda item: hashlib.sha256(
                            f"{int(seed)}:{query_id}:{int(item[0])}".encode("utf-8")
                        ).digest()
                    )
                    for target_track_id, support_images in eligible_tracks[: int(episodes_per_query)]:
                        ranked_supports = sorted(
                            support_images,
                            key=lambda image_id: (
                                -len(query_tracks.intersection(tracks_by_image.get(str(image_id), set()))),
                                hashlib.sha256(
                                    f"{int(seed)}:{query_id}:{int(target_track_id)}:{image_id}".encode("utf-8")
                                ).digest(),
                            ),
                        )
                        max_support_count = min(int(support_pairs), len(ranked_supports))
                        count_digest = hashlib.sha256(
                            f"{int(seed)}:{query_id}:{int(target_track_id)}:support_count".encode("utf-8")
                        ).digest()
                        desired_support_count = int(min_support_pairs) + int.from_bytes(
                            count_digest[:8],
                            byteorder="little",
                            signed=False,
                        ) % (max_support_count - int(min_support_pairs) + 1)
                        selected = ranked_supports[:desired_support_count]
                        if len(selected) < int(min_support_pairs):
                            continue
                        track_counts: dict[int, int] = {}
                        for image_id in selected:
                            for track_id in query_tracks.intersection(tracks_by_image.get(str(image_id), set())):
                                track_counts[int(track_id)] = int(track_counts.get(int(track_id), 0) + 1)
                        if int(track_counts.get(int(target_track_id), 0)) != len(selected):
                            raise RuntimeError("track-centric episode lost its target track in a selected support view")
                        episodes.append((query_id, tuple(selected), int(target_track_id)))
                        repeated_track_counts.append(int(sum(count >= 2 for count in track_counts.values())))
                        target_track_support_counts.append(int(len(selected)))
                        complete_track_counts.append(
                            int(sum(count == len(selected) for count in track_counts.values()))
                        )
                    continue
                candidate_counts: dict[str, int] = {}
                for track_id in query_tracks:
                    for image_id in images_by_track.get(int(track_id), []):
                        if str(image_id) == str(query_id) or str(image_id) not in effective_allowed_support_ids:
                            continue
                        candidate_counts[str(image_id)] = int(candidate_counts.get(str(image_id), 0) + 1)
                candidates = sorted(candidate_counts, key=lambda image_id: (-candidate_counts[image_id], image_id))[
                    : int(sfm_candidate_pool_size)
                ]
                if len(candidates) < int(min_support_pairs):
                    continue
                track_sets = {image_id: query_tracks.intersection(tracks_by_image[image_id]) for image_id in candidates}
                seen_support_sets: set[frozenset[str]] = set()
                for variant in range(min(int(episodes_per_query), len(candidates))):
                    selected = [candidates[int(variant)]]
                    track_counts: dict[int, int] = {int(track_id): 1 for track_id in track_sets[selected[0]]}
                    while len(selected) < int(support_pairs):
                        available = [image_id for image_id in candidates if image_id not in selected]
                        if not available:
                            break
                        best = max(
                            available,
                            key=lambda image_id: (
                                sum(
                                    10.0
                                    if int(track_counts.get(int(track_id), 0)) == 1
                                    else 1.0
                                    if int(track_counts.get(int(track_id), 0)) >= 2
                                    else 0.05
                                    for track_id in track_sets[image_id]
                                ),
                                len(track_sets[image_id]),
                                image_id,
                            ),
                        )
                        selected.append(best)
                        for track_id in track_sets[best]:
                            track_counts[int(track_id)] = int(track_counts.get(int(track_id), 0) + 1)
                    support_key = frozenset(selected)
                    if len(selected) < int(min_support_pairs) or support_key in seen_support_sets:
                        continue
                    seen_support_sets.add(support_key)
                    episodes.append((query_id, tuple(selected), None))
                    repeated_track_counts.append(int(sum(count >= 2 for count in track_counts.values())))
        if not episodes:
            raise ValueError("no multi-view landmark episodes could be constructed")
        self.provider = provider
        self.support_pairs = int(support_pairs)
        self.min_support_pairs = int(min_support_pairs)
        self.seed = int(seed)
        self.support_selection = str(support_selection)
        self.episodes_per_query = int(episodes_per_query)
        self.sfm_candidate_pool_size = int(sfm_candidate_pool_size)
        self.retrieval_tracks_per_episode = int(retrieval_tracks_per_episode)
        self.allowed_support_image_ids = (
            set(grouped)
            if allowed_support_image_ids is None
            else {str(image_id) for image_id in allowed_support_image_ids}
        )
        allowed_support_manifest_hash = hashlib.sha256(
            "\n".join(sorted(self.allowed_support_image_ids)).encode("utf-8")
        ).hexdigest()[:16]
        self.episodes = tuple(episodes)
        self.repeated_track_counts = tuple(repeated_track_counts)
        self.complete_track_counts = tuple(complete_track_counts)
        self.metadata = {
            **dict(getattr(provider, "metadata", {})),
            "episode_mode": "heldout_query_multi_view_track_prototype",
            "episode_count": int(len(self.episodes)),
            "episode_query_count": int(len({query_id for query_id, _indices, _track_id in self.episodes})),
            "episode_support_pairs": int(self.support_pairs),
            "episode_min_support_pairs": int(self.min_support_pairs),
            "episode_seed": int(self.seed),
            "episode_support_selection": str(self.support_selection),
            "episode_sfm_candidate_pool_size": int(self.sfm_candidate_pool_size),
            "episode_expected_repeated_track_mean": (
                0.0 if not self.repeated_track_counts else float(np.mean(self.repeated_track_counts))
            ),
            "episode_target_track_support_count_min": (
                0 if not target_track_support_counts else int(np.min(target_track_support_counts))
            ),
            "episode_target_track_support_count_mean": (
                0.0 if not target_track_support_counts else float(np.mean(target_track_support_counts))
            ),
            "episode_target_track_support_count_max": (
                0 if not target_track_support_counts else int(np.max(target_track_support_counts))
            ),
            "episode_target_track_support_count_sampling": (
                "deterministic_uniform_min_to_available_cap"
                if str(self.support_selection) == "sfm_track_episode"
                else "none"
            ),
            "episode_complete_track_count_mean": (
                0.0 if not complete_track_counts else float(np.mean(complete_track_counts))
            ),
            "episode_retrieval_track_cap": int(self.retrieval_tracks_per_episode),
            "episode_support_image_scope": (
                "train_query_images_only"
                if str(self.support_selection) in {"sfm_track_overlap", "sfm_track_episode"}
                else "manifest_records"
            ),
            "episode_allowed_support_image_count": int(len(self.allowed_support_image_ids)),
            "episode_allowed_support_manifest_hash": allowed_support_manifest_hash,
        }

    def __len__(self) -> int:
        return int(len(self.episodes))

    def get(self, index: int) -> MatchaJointTrainingSet:
        query_id, support_items, target_track_id = self.episodes[int(index)]
        if str(self.support_selection) in {"sfm_track_overlap", "sfm_track_episode"}:
            samples = [
                _materialize_index_only_joint_training_set(
                    self.provider.get_sfm_pair(str(query_id), str(reference_id))
                )
                for reference_id in support_items
            ]
        else:
            samples = [
                _materialize_index_only_joint_training_set(self.provider.get(int(record_index)))
                for record_index in support_items
            ]
        merged = merge_matcha_joint_training_sets(samples)
        pair_query_ids = np.asarray(merged.pair_query_ids, dtype=object).reshape(-1)
        if pair_query_ids.size == 0 or any(str(value) != str(query_id) for value in pair_query_ids.tolist()):
            raise ValueError("episodic provider merged records from different query images")
        if target_track_id is not None:
            selected_tracks = np.asarray(merged.landmark_track_ids, dtype=np.int64).reshape(-1)
            unique_tracks, support_counts = np.unique(selected_tracks, return_counts=True)
            complete_tracks = unique_tracks[support_counts == len(support_items)].astype(np.int64, copy=False)
            if int(target_track_id) not in set(complete_tracks.tolist()):
                raise ValueError(f"track-centric episode lost target track {target_track_id}")
            auxiliary_tracks = [
                int(track_id)
                for track_id in complete_tracks.tolist()
                if int(track_id) != int(target_track_id)
            ]
            auxiliary_tracks.sort(
                key=lambda track_id: hashlib.sha256(
                    f"{self.seed}:{query_id}:{int(target_track_id)}:{track_id}:retrieval".encode("utf-8")
                ).digest()
            )
            retrieval_track_ids = [int(target_track_id), *auxiliary_tracks][
                : int(self.retrieval_tracks_per_episode)
            ]
            target_keep = np.isin(selected_tracks, np.asarray(retrieval_track_ids, dtype=np.int64))
            support_count = int(np.count_nonzero(selected_tracks == int(target_track_id)))
            if support_count != len(support_items):
                raise ValueError(
                    f"track-centric episode target {target_track_id} has {support_count} support rows; "
                    f"expected {len(support_items)}"
                )
            merged = replace(
                merged,
                landmark_sample_pair_indices=np.asarray(merged.landmark_sample_pair_indices)[target_keep],
                landmark_query_xy=np.asarray(merged.landmark_query_xy)[target_keep],
                landmark_reference_xy=np.asarray(merged.landmark_reference_xy)[target_keep],
                landmark_track_ids=selected_tracks[target_keep],
                landmark_track_xyz=np.asarray(merged.landmark_track_xyz)[target_keep],
                landmark_support_view_counts=np.asarray(merged.landmark_support_view_counts)[target_keep],
            )
        return merged

    def landmark_retrieval_audit(self) -> dict[str, object]:
        output = dict(self.provider.landmark_retrieval_audit())
        support_counts = np.asarray([len(items) for _query_id, items, _track_id in self.episodes], dtype=np.int64)
        target_support_counts = np.asarray(
            [len(items) for _query_id, items, track_id in self.episodes if track_id is not None],
            dtype=np.int64,
        )
        output.update(
            {
                "episode_mode": "heldout_query_multi_view_track_prototype",
                "episode_count": int(len(self.episodes)),
                "episode_query_count": int(
                    len({query_id for query_id, _indices, _track_id in self.episodes})
                ),
                "episode_support_count_min": int(support_counts.min()),
                "episode_support_count_mean": float(support_counts.mean()),
                "episode_support_count_max": int(support_counts.max()),
                "query_excluded_from_support": True,
                "episode_support_selection": str(self.support_selection),
                "episode_expected_repeated_track_mean": (
                    0.0 if not self.repeated_track_counts else float(np.mean(self.repeated_track_counts))
                ),
                "episode_target_track_count": int(target_support_counts.size),
                "episode_target_track_support_count_min": (
                    0 if target_support_counts.size == 0 else int(target_support_counts.min())
                ),
                "episode_target_track_support_count_mean": (
                    0.0 if target_support_counts.size == 0 else float(target_support_counts.mean())
                ),
                "episode_target_track_support_count_max": (
                    0 if target_support_counts.size == 0 else int(target_support_counts.max())
                ),
                "episode_target_track_support_count_sampling": (
                    "deterministic_uniform_min_to_available_cap"
                    if str(self.support_selection) == "sfm_track_episode"
                    else "none"
                ),
                "episode_retrieval_track_cap": int(self.retrieval_tracks_per_episode),
                "episode_complete_track_count_mean": (
                    0.0 if not self.complete_track_counts else float(np.mean(self.complete_track_counts))
                ),
                "episode_support_image_scope": (
                    "train_query_images_only"
                    if str(self.support_selection) in {"sfm_track_overlap", "sfm_track_episode"}
                    else "manifest_records"
                ),
                "episode_allowed_support_image_count": int(len(self.allowed_support_image_ids)),
                "episode_allowed_support_manifest_hash": hashlib.sha256(
                    "\n".join(sorted(self.allowed_support_image_ids)).encode("utf-8")
                ).hexdigest()[:16],
            }
        )
        return output


def _initialize_distributed_runtime(args: argparse.Namespace) -> dict[str, int | bool | str]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if int(world_size) <= 1:
        return {
            "enabled": False,
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "backend": "",
        }
    if not torch.cuda.is_available():
        raise RuntimeError("distributed real RADIO training requires CUDA")
    local_rank = int(os.environ.get("LOCAL_RANK", str(args.local_rank)))
    if local_rank < 0 or local_rank >= int(torch.cuda.device_count()):
        raise ValueError(f"LOCAL_RANK={local_rank} is invalid for {torch.cuda.device_count()} CUDA devices")
    torch.cuda.set_device(int(local_rank))
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    rank = int(torch.distributed.get_rank())
    args.device = f"cuda:{local_rank}"
    return {
        "enabled": True,
        "rank": int(rank),
        "local_rank": int(local_rank),
        "world_size": int(torch.distributed.get_world_size()),
        "backend": str(torch.distributed.get_backend()),
    }


def _finish_distributed_runtime(runtime: dict[str, int | bool | str]) -> None:
    if not bool(runtime.get("enabled", False)):
        return
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def _validate_joint_localization_training_set(
    samples: MatchaJointTrainingSet,
    *,
    source: str,
    require_measurement_supervision: bool = False,
    require_landmark_retrieval_supervision: bool = False,
) -> dict[str, object]:
    missing: list[str] = []
    if samples.query_feature_maps is None or samples.render_feature_maps is None:
        missing.append("full-map query/reference feature maps")
    if samples.sample_pair_indices is None or samples.query_cell_indices is None or samples.render_cell_indices is None:
        missing.append("full-map correspondence pair/cell indices")
    if samples.query_rgb_images is None or samples.render_rgb_images is None:
        missing.append("RGB measurement images")
    if bool(require_measurement_supervision):
        fine_required = (
            samples.fine_sample_pair_indices,
            samples.fine_query_cell_indices,
            samples.fine_render_cell_indices,
            samples.fine_query_offset_labels,
            samples.fine_render_offset_labels,
        )
        if any(value is None for value in fine_required):
            missing.append("fine correspondence supervision for measurement loss")
    track_ids = None if samples.sample_track_ids is None else np.asarray(samples.sample_track_ids, dtype=np.int64)
    positive_mask = np.ones((samples.coarse_fine_samples.sample_count,), dtype=bool)
    if samples.sample_no_match_labels is not None:
        positive_mask &= np.asarray(samples.sample_no_match_labels, dtype=np.int64) == 0
    if samples.sample_ignore_mask is not None:
        positive_mask &= ~np.asarray(samples.sample_ignore_mask, dtype=bool)
    valid_positive_track_ids = (
        track_ids is not None
        and bool(np.any(positive_mask))
        and bool(np.all(track_ids[positive_mask] >= 0))
    )
    landmark_track_ids = (
        None
        if samples.landmark_track_ids is None
        else np.asarray(samples.landmark_track_ids, dtype=np.int64).reshape(-1)
    )
    valid_landmark_retrieval = (
        landmark_track_ids is not None
        and landmark_track_ids.size > 0
        and bool(np.all(landmark_track_ids >= 0))
        and samples.landmark_query_xy is not None
        and samples.landmark_reference_xy is not None
        and samples.pair_query_image_sizes is not None
        and samples.pair_reference_image_sizes is not None
    )
    if bool(require_landmark_retrieval_supervision) and not valid_landmark_retrieval:
        missing.append("continuous SfM observation supervision for query-to-landmark retrieval loss")
    if missing:
        raise ValueError(f"{source} joint cache is not a full-map real localization cache; missing {', '.join(missing)}")
    query_maps = np.asarray(samples.query_feature_maps, dtype=np.float32)
    reference_maps = np.asarray(samples.render_feature_maps, dtype=np.float32)
    query_rgb = np.asarray(samples.query_rgb_images, dtype=np.float32)
    reference_rgb = np.asarray(samples.render_rgb_images, dtype=np.float32)
    if int(query_maps.shape[0]) != int(reference_maps.shape[0]):
        raise ValueError(f"{source} query/reference feature maps must contain the same pair count")
    if int(query_maps.shape[1]) != int(reference_maps.shape[1]):
        raise ValueError(f"{source} query/reference feature-map channels must match")
    if int(query_maps.shape[1]) != int(samples.coarse_fine_samples.input_dim):
        raise ValueError(f"{source} feature-map channels must match coarse_fine_samples input_dim")
    if int(query_rgb.shape[0]) != int(query_maps.shape[0]) or int(reference_rgb.shape[0]) != int(reference_maps.shape[0]):
        raise ValueError(f"{source} RGB image count must match feature-map pair count")
    return {
        "source": str(source),
        "sample_count": int(samples.coarse_fine_samples.sample_count),
        "pair_count": int(query_maps.shape[0]),
        "input_dim": int(samples.coarse_fine_samples.input_dim),
        "query_feature_map_shape": [int(value) for value in query_maps.shape],
        "reference_feature_map_shape": [int(value) for value in reference_maps.shape],
        "query_rgb_shape": [int(value) for value in query_rgb.shape],
        "reference_rgb_shape": [int(value) for value in reference_rgb.shape],
        "has_measurement_fine_supervision": bool(samples.fine_sample_pair_indices is not None),
        "has_landmark_track_supervision": bool(valid_positive_track_ids),
        "landmark_track_supervision_count": int(0 if landmark_track_ids is None else landmark_track_ids.size),
        "landmark_xyz_supervision_count": int(
            0
            if samples.landmark_track_xyz is None
            else np.count_nonzero(np.isfinite(np.asarray(samples.landmark_track_xyz)).all(axis=1))
        ),
    }


def _load_first_manifest_shard(path: Path) -> tuple[MatchaJointTrainingSet, dict[str, object]]:
    manifest_path = Path(path)
    metadata = json.loads(manifest_path.read_text())
    shards = list(metadata.get("shards", []))
    if not shards:
        raise ValueError(f"{manifest_path} contains no joint-cache shards")
    shard_path = Path(str(shards[0]["path"]))
    if not shard_path.is_absolute():
        shard_path = manifest_path.parent / shard_path
    samples, _sample_metadata = load_matcha_joint_training_set_npz(shard_path)
    output_metadata = dict(metadata)
    output_metadata["audit_shard"] = str(shard_path)
    return samples, output_metadata


def _manifest_format(path: Path) -> str:
    return str(json.loads(Path(path).read_text()).get("format", ""))


def _is_referenced_manifest(path: Path) -> bool:
    return _manifest_format(Path(path)) == REFERENCED_MANIFEST_FORMAT


def _compact_referenced_metadata(metadata: dict[str, object]) -> dict[str, object]:
    output = dict(metadata)
    records = output.pop("records", None)
    if isinstance(records, list):
        output.setdefault("record_count", int(len(records)))
        output["records_preview"] = [
            {
                "query_id": str(item.get("query_id", "")),
                "reference_image_id": str(item.get("reference_image_id", "")),
                "row_count": int(item.get("row_count", len(item.get("row_indices", [])))),
            }
            for item in records[:3]
            if isinstance(item, dict)
        ]
    return output


def _load_referenced_joint_provider(
    path: Path,
    *,
    source: str,
    require_measurement_supervision: bool = False,
    require_landmark_retrieval_supervision: bool = False,
    feature_cache_size: int = 4,
    rgb_cache_size: int = 8,
    track_xyz_by_id: dict[int, np.ndarray] | None = None,
    track_observation_index: SfMTrackObservationIndex | None = None,
) -> tuple[RealRadioReferencedJointSampleProvider, MatchaJointTrainingSet, dict[str, object], dict[str, object]]:
    provider_kwargs: dict[str, object] = {
        "feature_cache_size": int(feature_cache_size),
        "rgb_cache_size": int(rgb_cache_size),
    }
    if track_xyz_by_id is not None:
        provider_kwargs["track_xyz_by_id"] = track_xyz_by_id
    if track_observation_index is not None:
        provider_kwargs["track_observation_index"] = track_observation_index
    provider = RealRadioReferencedJointSampleProvider(Path(path), **provider_kwargs)
    samples = provider.get(0)
    metadata = _compact_referenced_metadata(dict(getattr(provider, "metadata", json.loads(Path(path).read_text()))))
    metadata["audit_record_index"] = 0
    metadata["provider_sample_count"] = int(len(provider))
    audit = _validate_joint_localization_training_set(
        samples,
        source=str(source),
        require_measurement_supervision=bool(require_measurement_supervision),
        require_landmark_retrieval_supervision=bool(require_landmark_retrieval_supervision),
    )
    return provider, samples, metadata, audit


def _load_joint_set(
    path: Path,
    *,
    is_manifest: bool,
    source: str,
    require_measurement_supervision: bool = False,
    require_landmark_retrieval_supervision: bool = False,
) -> tuple[MatchaJointTrainingSet, dict[str, object], dict[str, object]]:
    if bool(is_manifest) and _is_referenced_manifest(Path(path)):
        _provider, samples, metadata, audit = _load_referenced_joint_provider(
            Path(path),
            source=str(source),
            require_measurement_supervision=bool(require_measurement_supervision),
            require_landmark_retrieval_supervision=bool(require_landmark_retrieval_supervision),
        )
        return samples, metadata, audit
    samples, metadata = (
        _load_first_manifest_shard(Path(path))
        if bool(is_manifest)
        else load_matcha_joint_training_set_npz(Path(path))
    )
    audit = _validate_joint_localization_training_set(
        samples,
        source=str(source),
        require_measurement_supervision=bool(require_measurement_supervision),
        require_landmark_retrieval_supervision=bool(require_landmark_retrieval_supervision),
    )
    return samples, dict(metadata), audit


def _resolve_radio_dual_dims(samples: MatchaJointTrainingSet, *, fine_input_dim: int, coarse_input_dim: int) -> tuple[int, int]:
    input_dim = int(samples.coarse_fine_samples.input_dim)
    fine = int(fine_input_dim)
    coarse = int(coarse_input_dim)
    if fine <= 0 and coarse <= 0:
        if input_dim % 2 != 0:
            raise ValueError("fine_input_dim/coarse_input_dim are required when input_dim is odd")
        return input_dim // 2, input_dim // 2
    if fine <= 0:
        fine = input_dim - coarse
    if coarse <= 0:
        coarse = input_dim - fine
    if fine <= 0 or coarse <= 0 or fine + coarse != input_dim:
        raise ValueError("fine_input_dim + coarse_input_dim must equal joint cache input_dim")
    return fine, coarse


def _build_config(args: argparse.Namespace, samples: MatchaJointTrainingSet) -> MatchaJointTrainingConfig:
    fine_dim, coarse_dim = (int(args.fine_input_dim), int(args.coarse_input_dim))
    if str(args.model_type) == "radio_dual_attention":
        fine_dim, coarse_dim = _resolve_radio_dual_dims(
            samples,
            fine_input_dim=int(args.fine_input_dim),
            coarse_input_dim=int(args.coarse_input_dim),
        )
    return MatchaJointTrainingConfig(
        model_type=str(args.model_type),
        output_dim=int(args.output_dim),
        residual_hidden_dim=int(args.residual_hidden_dim),
        fine_input_dim=int(fine_dim),
        coarse_input_dim=int(coarse_dim),
        attention_hidden_dim=int(args.attention_hidden_dim),
        attention_depth=int(args.attention_depth),
        attention_heads=int(args.attention_heads),
        attention_patch_size=int(args.attention_patch_size),
        attention_upsample_mode=str(args.attention_upsample_mode),
        attention_fusion_mode=str(args.attention_fusion_mode),
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        temperature=float(args.temperature),
        dual_softmax_weight=float(args.dual_softmax_weight),
        offset_loss_weight=float(args.offset_loss_weight),
        pair_fine_loss_weight=float(args.pair_fine_loss_weight),
        query_pair_fine_loss_weight=float(args.query_pair_fine_loss_weight),
        fine_continuous_loss_weight=float(args.fine_continuous_loss_weight),
        fine_loss_mode=str(args.fine_loss_mode),
        fine_uncertainty_loss_weight=float(args.fine_uncertainty_loss_weight),
        pair_confidence_loss_weight=float(args.pair_confidence_loss_weight),
        dense_heatmap_loss_weight=float(args.dense_heatmap_loss_weight),
        rgb_keypoint_loss_weight=float(args.rgb_keypoint_loss_weight),
        rgb_keypoint_position_loss_weight=float(args.rgb_keypoint_position_loss_weight),
        repeatability_loss_weight=float(args.repeatability_loss_weight),
        local_fine_transformer_loss_weight=float(args.local_fine_transformer_loss_weight),
        local_window_fine_loss_weight=float(args.local_window_fine_loss_weight),
        local_window_fine_mode=str(args.local_window_fine_mode),
        patch_correlation_loss_weight=float(args.patch_correlation_loss_weight),
        patch_correlation_window_size=int(args.patch_correlation_window_size),
        hard_negative_weight=float(args.hard_negative_weight),
        hard_negative_margin=float(args.hard_negative_margin),
        coarse_candidate_rank_loss_weight=float(args.coarse_candidate_rank_loss_weight),
        coarse_candidate_rank_margin=float(args.coarse_candidate_rank_margin),
        hard_false_match_weight=float(args.hard_false_match_weight),
        hard_false_match_margin=float(args.hard_false_match_margin),
        landmark_retrieval_loss_weight=float(args.landmark_retrieval_loss_weight),
        landmark_retrieval_temperature=float(args.landmark_retrieval_temperature),
        landmark_prototype_history_mix=float(args.landmark_prototype_history_mix),
        landmark_prototype_aggregation_method=str(args.landmark_prototype_aggregation_method),
        landmark_l2_normalize_observations=bool(args.landmark_l2_normalize_observations),
        landmark_normalize_final_prototypes=bool(args.landmark_normalize_final_prototypes),
        landmark_min_support_observations=int(args.landmark_min_support_observations),
        landmark_set_valued_cell_positives=bool(args.landmark_set_valued_cell_positives),
        landmark_memory_capacity=int(args.landmark_memory_capacity),
        landmark_frozen_negative_bank=str(args.landmark_frozen_negative_bank),
        landmark_memory_momentum=float(args.landmark_memory_momentum),
        landmark_memory_candidate_pool_size=int(args.landmark_memory_candidate_pool_size),
        landmark_semantic_hard_negatives_per_query=int(args.landmark_semantic_hard_negatives_per_query),
        landmark_geometry_hard_negatives_per_track=int(args.landmark_geometry_hard_negatives_per_track),
        landmark_random_negatives=int(args.landmark_random_negatives),
        landmark_max_memory_negatives=int(args.landmark_max_memory_negatives),
        landmark_dustbin_logit=float(args.landmark_dustbin_logit),
        landmark_dustbin_samples_per_image=int(args.landmark_dustbin_samples_per_image),
        landmark_dustbin_exclusion_radius_cells=int(args.landmark_dustbin_exclusion_radius_cells),
        landmark_dustbin_max_heatmap_target=float(args.landmark_dustbin_max_heatmap_target),
        landmark_dustbin_loss_weight=float(args.landmark_dustbin_loss_weight),
        landmark_dustbin_detach_descriptors=bool(args.landmark_dustbin_detach_descriptors),
        group_size=int(args.group_size),
        input_norm_mode=str(args.input_norm_mode),
        gate_mode=str(args.gate_mode),
        residual_gate_scale=float(args.residual_gate_scale),
        map_pair_batch_size=int(args.map_pair_batch_size),
        measurement_patch_loss_weight=float(args.measurement_patch_loss_weight),
        measurement_patch_direct_loss_weight=float(args.measurement_patch_direct_loss_weight),
        measurement_patch_epe_weight=float(args.measurement_patch_epe_weight),
        measurement_patch_dustbin_bce_weight=float(args.measurement_patch_dustbin_bce_weight),
        measurement_patch_batch_size=int(args.measurement_patch_batch_size),
        measurement_patch_max_samples_per_pair=int(args.measurement_patch_max_samples_per_pair),
        measurement_patch_search_radius_px=float(args.measurement_patch_search_radius_px),
        measurement_patch_context_radius_px=float(args.measurement_patch_context_radius_px),
        measurement_patch_step_px=float(args.measurement_patch_step_px),
        measurement_patch_coarse_search_radius_px=float(args.measurement_patch_coarse_search_radius_px),
        measurement_patch_coarse_step_px=float(args.measurement_patch_coarse_step_px),
        measurement_patch_feature_dim=int(args.measurement_patch_feature_dim),
        measurement_patch_hidden_dim=int(args.measurement_patch_hidden_dim),
        measurement_patch_target_heatmap_sigma_px=float(args.measurement_patch_target_heatmap_sigma_px),
        measurement_patch_dustbin_positive_weight=float(args.measurement_patch_dustbin_positive_weight),
        measurement_patch_encoder_arch=str(args.measurement_patch_encoder_arch),
        measurement_patch_input_mode=str(args.measurement_patch_input_mode),
        device=str(args.device),
        seed=int(args.seed),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--joint_cache", default="")
    source.add_argument("--joint_cache_manifest", default="")
    parser.add_argument("--validation_joint_cache", default="")
    parser.add_argument("--validation_joint_cache_manifest", default="")
    parser.add_argument("--warm_start_joint_checkpoint", default="")
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--output_joint_model", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--manifest_shard_cache_size", type=int, default=1)
    parser.add_argument("--manifest_steps_per_shard", type=int, default=1)
    parser.add_argument("--referenced_feature_cache_size", type=int, default=4)
    parser.add_argument("--referenced_rgb_cache_size", type=int, default=8)
    parser.add_argument("--landmark_track_observations", default="")
    parser.add_argument(
        "--landmark_track_observation_index_cache",
        default="",
        help="Validated NPZ cache for the full SfM per-image observation index.",
    )
    parser.add_argument(
        "--descriptor_token_manifest",
        default="",
        help="Token manifest used to record immutable RADIO source/preprocessing semantics in the checkpoint.",
    )
    parser.add_argument("--allow_sparse_landmark_csv_fallback", action="store_true")
    parser.add_argument("--provider_prefetch_workers", type=int, default=0)
    parser.add_argument("--provider_prefetch_depth", type=int, default=0)
    parser.add_argument("--provider_gradient_accumulation_pairs", type=int, default=1)
    parser.add_argument("--provider_pair_batch_size", type=int, default=1)
    parser.add_argument(
        "--landmark_episode_support_pairs",
        type=int,
        default=0,
        help="Group each held-out query with 2-8 distinct reference observations; 0 keeps pair-wise loading.",
    )
    parser.add_argument("--landmark_episode_min_support_pairs", type=int, default=2)
    parser.add_argument("--landmark_episode_seed", type=int, default=0)
    parser.add_argument(
        "--landmark_episode_support_selection",
        choices=("manifest", "sfm_track_overlap", "sfm_track_episode"),
        default="manifest",
    )
    parser.add_argument("--landmark_episodes_per_query", type=int, default=4)
    parser.add_argument("--landmark_episode_sfm_candidate_pool_size", type=int, default=32)
    parser.add_argument("--landmark_retrieval_tracks_per_episode", type=int, default=16)
    parser.add_argument("--provider_progress_interval_steps", type=int, default=0)
    parser.add_argument("--provider_empty_cuda_cache_interval_steps", type=int, default=0)
    parser.add_argument("--model_type", choices=("radio_dual_attention", "residual_adapter"), default="residual_adapter")
    parser.add_argument("--output_dim", type=int, default=128)
    parser.add_argument("--residual_hidden_dim", type=int, default=256)
    parser.add_argument("--fine_input_dim", type=int, default=0)
    parser.add_argument("--coarse_input_dim", type=int, default=0)
    parser.add_argument("--attention_hidden_dim", type=int, default=128)
    parser.add_argument("--attention_depth", type=int, default=1)
    parser.add_argument("--attention_heads", type=int, default=4)
    parser.add_argument("--attention_patch_size", type=int, default=4)
    parser.add_argument("--attention_upsample_mode", choices=("bilinear", "pixel_shuffle"), default="pixel_shuffle")
    parser.add_argument("--attention_fusion_mode", choices=("legacy", "matcha_original"), default="matcha_original")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--dual_softmax_weight", type=float, default=1.0)
    parser.add_argument("--offset_loss_weight", type=float, default=0.25)
    parser.add_argument("--pair_fine_loss_weight", type=float, default=0.0)
    parser.add_argument("--query_pair_fine_loss_weight", type=float, default=0.0)
    parser.add_argument("--fine_continuous_loss_weight", type=float, default=0.25)
    parser.add_argument("--fine_loss_mode", choices=("ce", "ce_plus_continuous", "continuous"), default="ce_plus_continuous")
    parser.add_argument("--fine_uncertainty_loss_weight", type=float, default=0.05)
    parser.add_argument("--pair_confidence_loss_weight", type=float, default=0.1)
    parser.add_argument("--dense_heatmap_loss_weight", type=float, default=0.25)
    parser.add_argument("--rgb_keypoint_loss_weight", type=float, default=0.0)
    parser.add_argument("--rgb_keypoint_position_loss_weight", type=float, default=0.0)
    parser.add_argument("--repeatability_loss_weight", type=float, default=0.0)
    parser.add_argument("--local_fine_transformer_loss_weight", type=float, default=0.0)
    parser.add_argument("--local_window_fine_loss_weight", type=float, default=0.5)
    parser.add_argument("--local_window_fine_mode", choices=("mlp", "correlation"), default="correlation")
    parser.add_argument("--patch_correlation_loss_weight", type=float, default=0.0)
    parser.add_argument("--patch_correlation_window_size", type=int, default=3)
    parser.add_argument("--hard_negative_weight", type=float, default=0.1)
    parser.add_argument("--hard_negative_margin", type=float, default=0.2)
    parser.add_argument("--coarse_candidate_rank_loss_weight", type=float, default=0.1)
    parser.add_argument("--coarse_candidate_rank_margin", type=float, default=0.2)
    parser.add_argument("--hard_false_match_weight", type=float, default=0.0)
    parser.add_argument("--hard_false_match_margin", type=float, default=0.2)
    parser.add_argument("--landmark_retrieval_loss_weight", type=float, default=0.25)
    parser.add_argument("--landmark_retrieval_temperature", type=float, default=0.07)
    parser.add_argument("--landmark_prototype_history_mix", type=float, default=0.5)
    parser.add_argument(
        "--landmark_prototype_aggregation_method",
        choices=("mean", "cosine_weighted_mean", "geometry_weighted"),
        default="mean",
    )
    parser.add_argument("--landmark_l2_normalize_observations", action="store_true")
    parser.add_argument(
        "--no_landmark_normalize_final_prototypes",
        dest="landmark_normalize_final_prototypes",
        action="store_false",
    )
    parser.set_defaults(landmark_normalize_final_prototypes=True)
    parser.add_argument("--landmark_min_support_observations", type=int, default=1)
    parser.add_argument(
        "--no_landmark_set_valued_cell_positives",
        dest="landmark_set_valued_cell_positives",
        action="store_false",
    )
    parser.set_defaults(landmark_set_valued_cell_positives=True)
    parser.add_argument("--landmark_memory_capacity", type=int, default=65536)
    parser.add_argument(
        "--landmark_frozen_negative_bank",
        default="",
        help="Read-only projected landmark snapshot shared by every DDP rank for global hard negatives.",
    )
    parser.add_argument(
        "--landmark_frozen_bank_support_observations",
        default="",
        help="Exact support-observation JSONL used to build the frozen bank; validated by hash.",
    )
    parser.add_argument("--landmark_memory_momentum", type=float, default=0.9)
    parser.add_argument("--landmark_memory_candidate_pool_size", type=int, default=4096)
    parser.add_argument("--landmark_semantic_hard_negatives_per_query", type=int, default=16)
    parser.add_argument("--landmark_geometry_hard_negatives_per_track", type=int, default=8)
    parser.add_argument("--landmark_random_negatives", type=int, default=128)
    parser.add_argument("--landmark_max_memory_negatives", type=int, default=2048)
    parser.add_argument("--landmark_dustbin_logit", type=float, default=0.0)
    parser.add_argument("--landmark_dustbin_samples_per_image", type=int, default=0)
    parser.add_argument("--landmark_dustbin_exclusion_radius_cells", type=int, default=1)
    parser.add_argument("--landmark_dustbin_max_heatmap_target", type=float, default=0.01)
    parser.add_argument("--landmark_dustbin_loss_weight", type=float, default=0.25)
    parser.add_argument(
        "--landmark_dustbin_backprop_descriptors",
        dest="landmark_dustbin_detach_descriptors",
        action="store_false",
        help="Ablation: allow no-match loss to update the global retrieval descriptor space.",
    )
    parser.set_defaults(landmark_dustbin_detach_descriptors=True)
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--input_norm_mode", choices=("identity", "layernorm"), default="identity")
    parser.add_argument("--gate_mode", choices=("residual", "sigmoid"), default="residual")
    parser.add_argument("--residual_gate_scale", type=float, default=0.1)
    parser.add_argument("--map_pair_batch_size", type=int, default=8)
    parser.add_argument("--measurement_patch_loss_weight", type=float, default=1.0)
    parser.add_argument("--measurement_patch_direct_loss_weight", type=float, default=0.25)
    parser.add_argument("--measurement_patch_epe_weight", type=float, default=0.05)
    parser.add_argument("--measurement_patch_dustbin_bce_weight", type=float, default=0.25)
    parser.add_argument("--measurement_patch_batch_size", type=int, default=128)
    parser.add_argument("--measurement_patch_max_samples_per_pair", type=int, default=512)
    parser.add_argument("--measurement_patch_search_radius_px", type=float, default=8.0)
    parser.add_argument("--measurement_patch_context_radius_px", type=float, default=8.0)
    parser.add_argument("--measurement_patch_step_px", type=float, default=1.0)
    parser.add_argument("--measurement_patch_coarse_search_radius_px", type=float, default=0.0)
    parser.add_argument("--measurement_patch_coarse_step_px", type=float, default=0.0)
    parser.add_argument("--measurement_patch_feature_dim", type=int, default=32)
    parser.add_argument("--measurement_patch_hidden_dim", type=int, default=64)
    parser.add_argument("--measurement_patch_target_heatmap_sigma_px", type=float, default=0.5)
    parser.add_argument("--measurement_patch_dustbin_positive_weight", type=float, default=1.0)
    parser.add_argument("--measurement_patch_encoder_arch", default="simple")
    parser.add_argument("--measurement_patch_input_mode", default="rgb")
    parser.add_argument("--validation_interval", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    distributed_runtime = _initialize_distributed_runtime(args)
    started = time.perf_counter()
    train_path = Path(args.joint_cache_manifest or args.joint_cache)
    require_measurement = float(args.measurement_patch_loss_weight) > 0.0
    require_landmark_retrieval = float(args.landmark_retrieval_loss_weight) > 0.0
    track_observation_index = (
        load_track_observation_index(
            Path(args.landmark_track_observations),
            cache_path=(
                Path(args.landmark_track_observation_index_cache)
                if str(args.landmark_track_observation_index_cache)
                else None
            ),
        )
        if str(args.landmark_track_observations)
        else None
    )
    track_xyz_by_id = (
        None if track_observation_index is None else track_observation_index.track_xyz_by_id
    )
    train_provider = None
    train_is_referenced_manifest = bool(args.joint_cache_manifest) and _is_referenced_manifest(train_path)
    if train_is_referenced_manifest:
        train_provider, train_samples, train_metadata, train_audit = _load_referenced_joint_provider(
            train_path,
            source="train",
            require_measurement_supervision=bool(require_measurement),
            require_landmark_retrieval_supervision=bool(require_landmark_retrieval),
            feature_cache_size=int(args.referenced_feature_cache_size),
            rgb_cache_size=int(args.referenced_rgb_cache_size),
            track_xyz_by_id=track_xyz_by_id,
            track_observation_index=track_observation_index,
        )
    else:
        train_samples, train_metadata, train_audit = _load_joint_set(
            train_path,
            is_manifest=bool(args.joint_cache_manifest),
            source="train",
            require_measurement_supervision=bool(require_measurement),
            require_landmark_retrieval_supervision=bool(require_landmark_retrieval),
        )
    if train_provider is not None and bool(require_landmark_retrieval) and int(args.landmark_episode_support_pairs) > 0:
        train_provider = RealRadioMultiViewEpisodeProvider(
            train_provider,
            support_pairs=int(args.landmark_episode_support_pairs),
            min_support_pairs=int(args.landmark_episode_min_support_pairs),
            seed=int(args.landmark_episode_seed),
            support_selection=str(args.landmark_episode_support_selection),
            episodes_per_query=int(args.landmark_episodes_per_query),
            sfm_candidate_pool_size=int(args.landmark_episode_sfm_candidate_pool_size),
            retrieval_tracks_per_episode=int(args.landmark_retrieval_tracks_per_episode),
        )
        train_samples = train_provider.get(0)
        train_metadata.update(
            {
                "episode_mode": "heldout_query_multi_view_track_prototype",
                "episode_count": int(len(train_provider)),
                "episode_support_pairs": int(args.landmark_episode_support_pairs),
                "episode_min_support_pairs": int(args.landmark_episode_min_support_pairs),
                "episode_support_selection": str(args.landmark_episode_support_selection),
            }
        )
        train_audit = _validate_joint_localization_training_set(
            train_samples,
            source="train_episode",
            require_measurement_supervision=bool(require_measurement),
            require_landmark_retrieval_supervision=True,
        )
    if train_provider is not None and bool(require_landmark_retrieval):
        dataset_audit = train_provider.landmark_retrieval_audit()
        if (
            str(dataset_audit.get("supervision_source", "")) != "sfm_common_track_observations"
            and not bool(args.allow_sparse_landmark_csv_fallback)
        ):
            raise ValueError(
                "landmark retrieval requires full SfM common-track observations; pass "
                "--landmark_track_observations or explicitly allow the sparse CSV diagnostic fallback"
            )
        dataset_audit["memory_capacity"] = int(args.landmark_memory_capacity)
        dataset_audit["memory_capacity_covers_unique_tracks"] = bool(
            int(args.landmark_memory_capacity) >= int(dataset_audit["unique_track_count"])
        )
        train_audit["landmark_retrieval_dataset"] = dataset_audit
    validation_samples = None
    validation_provider = None
    validation_manifest_path = None
    validation_metadata: dict[str, object] = {}
    validation_audit: dict[str, object] = {}
    validation_path = Path(args.validation_joint_cache_manifest or args.validation_joint_cache) if (args.validation_joint_cache_manifest or args.validation_joint_cache) else None
    if validation_path is not None:
        validation_is_referenced = bool(args.validation_joint_cache_manifest) and _is_referenced_manifest(validation_path)
        if validation_is_referenced:
            validation_provider, validation_samples, validation_metadata, validation_audit = _load_referenced_joint_provider(
                validation_path,
                source="validation",
                require_measurement_supervision=bool(require_measurement),
                require_landmark_retrieval_supervision=bool(require_landmark_retrieval),
                feature_cache_size=int(args.referenced_feature_cache_size),
                rgb_cache_size=int(args.referenced_rgb_cache_size),
                track_xyz_by_id=track_xyz_by_id,
                track_observation_index=track_observation_index,
            )
            if bool(require_landmark_retrieval) and int(args.landmark_episode_support_pairs) > 0:
                validation_provider = RealRadioMultiViewEpisodeProvider(
                    validation_provider,
                    support_pairs=int(args.landmark_episode_support_pairs),
                    min_support_pairs=int(args.landmark_episode_min_support_pairs),
                    seed=int(args.landmark_episode_seed) + 1,
                    support_selection=str(args.landmark_episode_support_selection),
                    episodes_per_query=int(args.landmark_episodes_per_query),
                    sfm_candidate_pool_size=int(args.landmark_episode_sfm_candidate_pool_size),
                    retrieval_tracks_per_episode=int(args.landmark_retrieval_tracks_per_episode),
                    allowed_support_image_ids=getattr(train_provider, "allowed_support_image_ids", None),
                )
                validation_samples = validation_provider.get(0)
                validation_metadata.update(
                    {
                        "episode_mode": "heldout_query_multi_view_track_prototype",
                        "episode_count": int(len(validation_provider)),
                        "episode_support_pairs": int(args.landmark_episode_support_pairs),
                    }
                )
                validation_audit = _validate_joint_localization_training_set(
                    validation_samples,
                    source="validation_episode",
                    require_measurement_supervision=bool(require_measurement),
                    require_landmark_retrieval_supervision=True,
                )
        else:
            validation_samples, validation_metadata, validation_audit = _load_joint_set(
                validation_path,
                is_manifest=bool(args.validation_joint_cache_manifest),
                source="validation",
                require_measurement_supervision=bool(require_measurement),
                require_landmark_retrieval_supervision=bool(require_landmark_retrieval),
            )
        if bool(args.validation_joint_cache_manifest) and not validation_is_referenced:
            validation_manifest_path = validation_path
            validation_samples = None
    frozen_bank_contract: dict[str, object] = {}
    if str(args.landmark_frozen_negative_bank):
        support_observations = Path(
            args.landmark_frozen_bank_support_observations or args.landmark_track_observations
        )
        if not str(support_observations):
            raise ValueError("frozen landmark bank validation requires support observations")
        frozen_bank_contract = validate_frozen_landmark_bank_contract(
            Path(args.landmark_frozen_negative_bank),
            warm_start_checkpoint=(
                Path(args.warm_start_joint_checkpoint) if str(args.warm_start_joint_checkpoint) else None
            ),
            support_observations=support_observations,
            expected_source_image_count=(
                len(train_provider.allowed_support_image_ids)
                if isinstance(train_provider, RealRadioMultiViewEpisodeProvider)
                and str(train_provider.support_selection) in {"sfm_track_overlap", "sfm_track_episode"}
                else None
            ),
        )
    cfg = _build_config(args, train_samples)
    descriptor_source_config: dict[str, object] = {}
    if str(args.descriptor_token_manifest):
        descriptor_manifest = TokenBankManifest.from_json(Path(args.descriptor_token_manifest))
        descriptor_manifest.validate(verify_checksums=False)
        feature_key = str(train_metadata.get("feature_key", "radio_final"))
        descriptor_source_config = token_feature_source_config(descriptor_manifest, feature_key)
        if int(descriptor_source_config["input_channels"]) != int(train_samples.coarse_fine_samples.input_dim):
            raise ValueError(
                "descriptor token source does not match training input dimension: "
                f"manifest={descriptor_source_config['input_channels']!r}, "
                f"training={train_samples.coarse_fine_samples.input_dim!r}"
            )
    warm_start_model = None
    if str(args.warm_start_joint_checkpoint):
        warm_start_model = load_matcha_joint_model(Path(args.warm_start_joint_checkpoint), device=str(args.device)).model
    if train_provider is not None:
        run = train_matcha_joint_model_from_sample_provider(
            int(len(train_provider)),
            train_provider.get,
            cfg,
            validation_sample_count=0 if validation_provider is None else int(len(validation_provider)),
            get_validation_sample=None if validation_provider is None else validation_provider.get,
            validation_interval=int(args.validation_interval),
            steps_per_sample=int(args.manifest_steps_per_shard),
            provider_gradient_accumulation_pairs=int(args.provider_gradient_accumulation_pairs),
            provider_pair_batch_size=int(args.provider_pair_batch_size),
            provider_prefetch_workers=int(args.provider_prefetch_workers),
            provider_prefetch_depth=int(args.provider_prefetch_depth),
            provider_progress_interval_steps=int(args.provider_progress_interval_steps),
            provider_empty_cuda_cache_interval_steps=int(args.provider_empty_cuda_cache_interval_steps),
            warm_start_model=warm_start_model,
            provider_name=(
                "real_radio_multiview_landmark_episode"
                if isinstance(train_provider, RealRadioMultiViewEpisodeProvider)
                else "real_radio_referenced_manifest"
            ),
        )
    elif bool(args.joint_cache_manifest):
        if bool(distributed_runtime["enabled"]):
            raise ValueError("distributed training is currently supported only for image-referenced lazy manifests")
        run = train_matcha_joint_model_from_manifest(
            train_path,
            cfg,
            validation_samples=validation_samples,
            validation_manifest_path=validation_manifest_path,
            validation_interval=int(args.validation_interval),
            shard_cache_size=int(args.manifest_shard_cache_size),
            steps_per_shard=int(args.manifest_steps_per_shard),
            warm_start_model=warm_start_model,
        )
    else:
        if bool(distributed_runtime["enabled"]):
            raise ValueError("distributed training is currently supported only for image-referenced lazy manifests")
        run = train_matcha_joint_model(
            train_samples,
            cfg,
            validation_samples=validation_samples,
            validation_interval=int(args.validation_interval),
            warm_start_model=warm_start_model,
        )
    if int(distributed_runtime["rank"]) != 0:
        _finish_distributed_runtime(distributed_runtime)
        return
    prototype_builder = TrackPrototypeBuilder(
        aggregation=LandmarkAggregationConfig(
            method=str(cfg.landmark_prototype_aggregation_method),
            min_observations=max(1, int(cfg.landmark_min_support_observations)),
            l2_normalize_observations=bool(cfg.landmark_l2_normalize_observations),
        ),
        normalize_final_prototypes=bool(cfg.landmark_normalize_final_prototypes),
    )
    run.summary.update(
        {
            "descriptor_source_config": descriptor_source_config,
            "descriptor_token_manifest": str(args.descriptor_token_manifest),
            "track_prototype_builder": prototype_builder.to_dict(),
            "frozen_landmark_bank_contract": frozen_bank_contract,
            "landmark_episode_support_pairs": int(args.landmark_episode_support_pairs),
            "landmark_episode_min_support_pairs": int(args.landmark_episode_min_support_pairs),
        }
    )
    adapter_run = joint_run_as_coarse_fine_adapter_run(run)
    save_matcha_coarse_fine_adapter(adapter_run, Path(args.output_model))
    save_matcha_joint_model(run, Path(args.output_joint_model))
    summary = {
        "stage": "real_radio_joint_localization_training",
        "elapsed_sec": float(time.perf_counter() - started),
        "joint_cache": {
            "path": str(train_path),
            "is_manifest": bool(args.joint_cache_manifest),
            "metadata": train_metadata,
            "audit": train_audit,
        },
        "validation_joint_cache": {
            "path": "" if validation_path is None else str(validation_path),
            "is_manifest": bool(args.validation_joint_cache_manifest),
            "metadata": validation_metadata,
            "audit": validation_audit,
        },
        "joint_training_contract": {
            "requires_full_feature_maps": True,
            "requires_full_map_correspondence_indices": True,
            "requires_fine_measurement_supervision": bool(require_measurement),
            "requires_sfm_track_identity": bool(require_landmark_retrieval),
            "landmark_retrieval_target": "query_full_map_to_multi_observation_track_prototype",
            "landmark_hard_negative_sources": ["same_pair_covisible", "global_semantic_memory", "nearby_3d_memory"],
            "landmark_track_observations": str(args.landmark_track_observations),
            "requires_rgb_measurement_images": True,
            "rejects_row_only_sample_cache": True,
            "reference_source": "real_image",
            "default_feature_source": "radio_final",
            "descriptor_source_config": descriptor_source_config,
            "track_prototype_builder": prototype_builder.to_dict(),
        },
        "distributed_runtime": dict(distributed_runtime),
        "frozen_landmark_bank_contract": frozen_bank_contract,
        "config": asdict(cfg),
        "training": dict(run.summary),
        "outputs": {
            "adapter_model": str(args.output_model),
            "joint_model": str(args.output_joint_model),
            "summary": str(args.summary_json),
        },
    }
    output = Path(args.summary_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    _finish_distributed_runtime(distributed_runtime)


if __name__ == "__main__":  # pragma: no cover
    main()
