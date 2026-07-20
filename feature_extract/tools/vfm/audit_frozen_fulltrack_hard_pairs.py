"""Audit raw full-track appearance on frozen rank-2-to-20 hard pairs.

The full-track exporter is target-free.  This evaluation-only program joins
registered SfM identities afterwards and asks a narrower question than global
candidate AP: when the frozen mapper prior ranks a wrong track above a correct
top-20 track, does an appearance feature prefer the correct track?  A subset
of those pairs is additionally marked as a coherent 3-D shift when a query
contains several wrong-minus-correct vectors with one stable displacement.

The result is diagnostic only.  It neither fits a calibrator nor emits an
inference or pose-scoring input.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.tools.vfm.audit_frozen_fulltrack_candidate_appearance import (
    merge_frozen_fulltrack_appearance,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import read_colmap_images_binary
from feature_extract.vfm.localization.landmark_hybrid import load_landmark_index_npz
from feature_extract.vfm.localization.query_observation_identity import (
    registered_candidate_identity_labels,
    registered_query_observation_targets,
)


ARTIFACT_FORMAT = "frozen_fulltrack_hard_pair_audit_v1"


@dataclass(frozen=True)
class HardPairSelection:
    """One correct-versus-frozen-top-wrong comparison per hard query point."""

    row_indices: np.ndarray
    positive_columns: np.ndarray
    negative_columns: np.ndarray
    positive_ranks: np.ndarray
    query_ids: np.ndarray
    wrong_minus_correct_xyz: np.ndarray

    def __post_init__(self) -> None:
        rows = np.asarray(self.row_indices, dtype=np.int64).reshape(-1)
        positive = np.asarray(self.positive_columns, dtype=np.int64).reshape(-1)
        negative = np.asarray(self.negative_columns, dtype=np.int64).reshape(-1)
        ranks = np.asarray(self.positive_ranks, dtype=np.int64).reshape(-1)
        query_ids = np.asarray(self.query_ids).astype(str).reshape(-1)
        deltas = np.asarray(self.wrong_minus_correct_xyz, dtype=np.float64)
        count = len(rows)
        if (
            np.any(rows < 0)
            or np.any(positive < 0)
            or np.any(negative < 0)
            or np.any(positive == negative)
            or np.any(ranks < 2)
            or len(query_ids) != count
            or deltas.shape != (count, 3)
            or np.any(~np.isfinite(deltas))
        ):
            raise ValueError("hard-pair selection arrays are invalid")
        object.__setattr__(self, "row_indices", rows)
        object.__setattr__(self, "positive_columns", positive)
        object.__setattr__(self, "negative_columns", negative)
        object.__setattr__(self, "positive_ranks", ranks)
        object.__setattr__(self, "query_ids", query_ids)
        object.__setattr__(self, "wrong_minus_correct_xyz", deltas)

    @property
    def pair_count(self) -> int:
        return int(len(self.row_indices))


def _paths(value: str) -> tuple[Path, ...]:
    paths = tuple(Path(item.strip()) for item in str(value).split(",") if item.strip())
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("appearance artifact paths must be non-empty and unique")
    return paths


def _splits(value: str) -> tuple[str, ...]:
    splits = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not splits or len(set(splits)) != len(splits):
        raise ValueError("audit splits must be non-empty and unique")
    if set(splits) - {"train", "validation", "test"}:
        raise ValueError("audit splits contain an unsupported name")
    return splits


def _metadata(data: Mapping[str, np.ndarray], *, context: str) -> dict[str, Any]:
    if "metadata_json" not in data:
        raise ValueError(f"{context} lacks metadata_json")
    value = json.loads(str(np.asarray(data["metadata_json"]).item()))
    if not isinstance(value, dict):
        raise ValueError(f"{context} metadata is not an object")
    return value


def _array_sha256_short(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()[:16]


def _typed_array_sha256_short(values: np.ndarray) -> str:
    """Hash dtype, shape, and bytes for newer full-track CSR children."""

    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()[:16]


def _declared_array_hash_matches(
    values: np.ndarray,
    *,
    declared_hash: str,
    scheme: str,
) -> bool:
    """Accept only explicitly supported legacy or typed CSR hash contracts.

    Older frozen children recorded a raw-byte digest.  Newer exporters record
    dtype and shape as well, preventing a same-byte layout collision.  An
    absent scheme is kept backward compatible, but it still has to match one
    of those two exact digests.
    """

    normalized_scheme = str(scheme).strip()
    legacy = _array_sha256_short(values)
    typed = _typed_array_sha256_short(values)
    if normalized_scheme in {"", "legacy_raw_bytes_sha256_v1"}:
        return str(declared_hash) in {legacy, typed}
    if normalized_scheme == "dtype_shape_bytes_sha256_v1":
        return str(declared_hash) == typed
    raise ValueError("full-track child declares an unsupported CSR hash scheme")


def _resolve_workspace_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else Path.cwd() / path


def _validate_projected_bank_lineage(
    metadata: Sequence[Mapping[str, Any]], *, projected_landmark_bank: Path
) -> None:
    """Bind full-track summaries to the exact projected landmark bank for XYZ."""

    expected_hash = file_sha256_short(projected_landmark_bank)
    for item in metadata:
        source_path_value = item.get("source_frozen_appearance_artifact")
        source_hash = str(item.get("source_frozen_appearance_artifact_sha256", ""))
        # Both aligned-layout and global-context summaries may be derived from
        # a current raw per-view CSR artifact.  Always verify that bridge when
        # it is declared, even if the child also copied the S0 root fields.
        # Otherwise a stale or different raw CSR handoff could masquerade as a
        # valid summary merely because it references the same landmark bank.
        raw_path_value = item.get("source_fulltrack_per_view_artifact")
        raw_hash = str(item.get("source_fulltrack_per_view_artifact_sha256", ""))
        if not raw_path_value:
            raw_path_value = item.get("source_frozen_fulltrack_per_view_artifact")
            raw_hash = str(
                item.get("source_frozen_fulltrack_per_view_artifact_sha256", "")
            )
        if raw_path_value:
            raw_path = _resolve_workspace_path(raw_path_value)
            if (
                not raw_hash
                or not raw_path.is_file()
                or file_sha256_short(raw_path) != raw_hash
            ):
                raise ValueError("full-track raw per-view source lineage is stale")
            with np.load(raw_path, allow_pickle=False) as raw_source:
                raw_metadata = _metadata(
                    raw_source, context="full-track raw per-view source"
                )
                declared_arrays = {
                    "source_edge_candidate_offsets_sha256": "edge_candidate_offsets",
                    # The first aligned-layout exporter used the shorter key.
                    # Treat it as a strict alias rather than leaving offsets
                    # outside the CSR lineage check.
                    "source_edge_offsets_sha256": "edge_candidate_offsets",
                    "source_edge_geometry_rows_sha256": "edge_geometry_rows",
                    "source_candidate_tracks_sha256": "candidate_track_ids",
                    "source_candidate_probabilities_sha256": "candidate_probabilities",
                    "source_null_probabilities_sha256": "null_probabilities",
                    "source_verification_rows_sha256": "verification_source_row_indices",
                }
                hash_scheme = str(item.get("source_csr_array_hash_scheme", ""))
                for metadata_key, raw_field in declared_arrays.items():
                    declared_hash = str(item.get(metadata_key, "")).strip()
                    if not declared_hash:
                        continue
                    if raw_field not in raw_source.files:
                        raise ValueError(
                            "full-track child declares a raw CSR array absent from "
                            f"its source: {raw_field}"
                        )
                    if not _declared_array_hash_matches(
                        raw_source[raw_field],
                        declared_hash=declared_hash,
                        scheme=hash_scheme,
                    ):
                        raise ValueError(
                            "full-track child raw CSR/candidate lineage differs for "
                            f"{raw_field}"
                        )
            if (
                raw_metadata.get("format")
                != "frozen_fulltrack_candidate_per_view_appearance_v1"
                or raw_metadata.get("per_view_edge_feature_semantics")
                != "raw_aligned_ncc_per_real_sfm_observation_v1"
            ):
                raise ValueError("summary source is not the immutable raw per-view path")
            declared_semantics = str(item.get("source_edge_feature_semantics", "")).strip()
            if declared_semantics and declared_semantics != str(
                raw_metadata.get("per_view_edge_feature_semantics", "")
            ):
                raise ValueError("full-track child raw edge semantics differ")
            declared_contract = str(item.get("source_fulltrack_edge_contract", "")).strip()
            if declared_contract and declared_contract != "raw_fulltrack_per_view_csr_v1":
                raise ValueError("full-track child has an unsupported raw CSR contract")
            raw_source_path = raw_metadata.get("source_frozen_appearance_artifact")
            raw_source_hash = str(
                raw_metadata.get("source_frozen_appearance_artifact_sha256", "")
            )
            if source_path_value and (
                str(source_path_value) != str(raw_source_path)
                or source_hash != raw_source_hash
            ):
                raise ValueError("summary S0 root differs from its raw per-view source")
            source_path_value = raw_source_path
            source_hash = raw_source_hash
        source_path = _resolve_workspace_path(source_path_value)
        if (
            not source_hash
            or not source_path.is_file()
            or file_sha256_short(source_path) != source_hash
        ):
            raise ValueError("full-track source frozen appearance lineage is stale")
        with np.load(source_path, allow_pickle=False) as source:
            source_metadata = _metadata(source, context="source frozen appearance artifact")
        inputs = source_metadata.get("inputs")
        if not isinstance(inputs, Mapping):
            raise ValueError("source frozen appearance artifact lacks input lineage")
        bank = inputs.get("projected_landmark_bank")
        if not isinstance(bank, Mapping) or str(bank.get("sha256", "")) != expected_hash:
            raise ValueError("hard-pair bank differs from frozen appearance source")


def select_rank2_to_top1_wrong_pairs(
    *,
    query_ids: np.ndarray,
    candidate_track_ids: np.ndarray,
    candidate_probabilities: np.ndarray,
    correct_labels: np.ndarray,
    xyz_by_track: Mapping[int, np.ndarray],
) -> HardPairSelection:
    """Choose a target-side correct-vs-top-wrong pair without changing top-L.

    The target-side positive is the highest-prior registered correct candidate.
    It is retained only when its stable frozen-prior rank is at least two; the
    negative is the highest-prior non-positive candidate.  No visual score is
    read in this selection, so every feature sees exactly the same hard pairs.
    """

    ids = np.asarray(query_ids).astype(str).reshape(-1)
    tracks = np.asarray(candidate_track_ids, dtype=np.int64)
    probabilities = np.asarray(candidate_probabilities, dtype=np.float64)
    labels = np.asarray(correct_labels, dtype=bool)
    if (
        tracks.ndim != 2
        or probabilities.shape != tracks.shape
        or labels.shape != tracks.shape
        or len(ids) != len(tracks)
        or np.any(~np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
    ):
        raise ValueError("hard-pair candidate inputs are incompatible")
    rows: list[int] = []
    positive_columns: list[int] = []
    negative_columns: list[int] = []
    ranks: list[int] = []
    selected_ids: list[str] = []
    deltas: list[np.ndarray] = []
    for row in range(len(ids)):
        valid = (tracks[row] >= 0) & (probabilities[row] > 0.0)
        positive = valid & labels[row]
        negative = valid & ~labels[row]
        if not np.any(positive) or not np.any(negative):
            continue
        ordered_columns = np.flatnonzero(valid)[
            np.argsort(-probabilities[row, valid], kind="stable")
        ]
        positive_column = int(
            np.flatnonzero(positive)[
                np.argmax(probabilities[row, np.flatnonzero(positive)])
            ]
        )
        positive_position = np.flatnonzero(ordered_columns == positive_column)
        if len(positive_position) != 1:
            raise RuntimeError("hard-pair positive is absent from its frozen ordering")
        rank = int(positive_position[0] + 1)
        if rank < 2:
            continue
        negative_column = int(next(column for column in ordered_columns if negative[column]))
        positive_track = int(tracks[row, positive_column])
        negative_track = int(tracks[row, negative_column])
        if positive_track not in xyz_by_track or negative_track not in xyz_by_track:
            continue
        delta = np.asarray(xyz_by_track[negative_track], dtype=np.float64) - np.asarray(
            xyz_by_track[positive_track], dtype=np.float64
        )
        if delta.shape != (3,) or np.any(~np.isfinite(delta)):
            continue
        rows.append(row)
        positive_columns.append(positive_column)
        negative_columns.append(negative_column)
        ranks.append(rank)
        selected_ids.append(str(ids[row]))
        deltas.append(delta)
    return HardPairSelection(
        row_indices=np.asarray(rows, dtype=np.int64),
        positive_columns=np.asarray(positive_columns, dtype=np.int64),
        negative_columns=np.asarray(negative_columns, dtype=np.int64),
        positive_ranks=np.asarray(ranks, dtype=np.int64),
        query_ids=np.asarray(selected_ids),
        wrong_minus_correct_xyz=np.asarray(deltas, dtype=np.float64).reshape(-1, 3),
    )


def coherent_shift_mask(
    selection: HardPairSelection,
    *,
    minimum_pairs: int,
    minimum_shift_m: float,
    maximum_dispersion_m: float,
    maximum_relative_dispersion: float,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Mark query groups whose top-wrong alternatives have one stable 3-D shift."""

    if (
        int(minimum_pairs) <= 0
        or float(minimum_shift_m) < 0.0
        or float(maximum_dispersion_m) < 0.0
        or float(maximum_relative_dispersion) < 0.0
    ):
        raise ValueError("coherent-shift thresholds are invalid")
    mask = np.zeros((selection.pair_count,), dtype=bool)
    diagnostics: list[dict[str, object]] = []
    for query_id in dict.fromkeys(selection.query_ids.tolist()):
        indices = np.flatnonzero(selection.query_ids == query_id)
        deltas = selection.wrong_minus_correct_xyz[indices]
        center = np.median(deltas, axis=0)
        shift_norm = float(np.linalg.norm(center))
        dispersion = float(np.median(np.linalg.norm(deltas - center, axis=1)))
        limit = max(
            float(maximum_dispersion_m),
            float(maximum_relative_dispersion) * shift_norm,
        )
        coherent = bool(
            len(indices) >= int(minimum_pairs)
            and shift_norm >= float(minimum_shift_m)
            and dispersion <= limit
        )
        if coherent:
            mask[indices] = True
        diagnostics.append(
            {
                "query_id": str(query_id),
                "pair_count": int(len(indices)),
                "shift_norm_m": shift_norm,
                "shift_dispersion_m": dispersion,
                "coherent": coherent,
            }
        )
    return mask, diagnostics


def wilson_lower_bound(success_count: int, trial_count: int, *, z: float = 1.959963984540054) -> float | None:
    """Return a deterministic two-sided 95% Wilson lower confidence bound."""

    successes = int(success_count)
    trials = int(trial_count)
    if trials <= 0:
        return None
    if successes < 0 or successes > trials or not np.isfinite(float(z)) or float(z) <= 0.0:
        raise ValueError("Wilson interval inputs are invalid")
    proportion = successes / trials
    squared = float(z) ** 2
    denominator = 1.0 + squared / trials
    center = proportion + squared / (2.0 * trials)
    radius = float(z) * math.sqrt(
        proportion * (1.0 - proportion) / trials + squared / (4.0 * trials**2)
    )
    return float((center - radius) / denominator)


def pairwise_feature_metrics(
    *,
    scores: np.ndarray,
    feature_valid: np.ndarray,
    selection: HardPairSelection,
    subset_mask: np.ndarray | None = None,
) -> dict[str, object]:
    """Score one raw feature on a fixed selection of target-side hard pairs."""

    values = np.asarray(scores, dtype=np.float64)
    valid = np.asarray(feature_valid, dtype=bool)
    if values.shape != valid.shape or values.ndim != 2:
        raise ValueError("pairwise feature scores and masks are incompatible")
    subset = (
        np.ones((selection.pair_count,), dtype=bool)
        if subset_mask is None
        else np.asarray(subset_mask, dtype=bool).reshape(-1)
    )
    if subset.shape != (selection.pair_count,):
        raise ValueError("pairwise subset mask is incompatible")
    rows = selection.row_indices[subset]
    positive = selection.positive_columns[subset]
    negative = selection.negative_columns[subset]
    total = int(len(rows))
    if total == 0:
        return {
            "pair_count": 0,
            "usable_pair_count": 0,
            "usable_pair_rate": None,
            "win_count": 0,
            "tie_count": 0,
            "loss_count": 0,
            "win_rate": None,
            "win_rate_wilson95_lower": None,
            "median_correct_minus_wrong": None,
            "p10_correct_minus_wrong": None,
        }
    usable = (
        valid[rows, positive]
        & valid[rows, negative]
        & np.isfinite(values[rows, positive])
        & np.isfinite(values[rows, negative])
    )
    differences = values[rows[usable], positive[usable]] - values[rows[usable], negative[usable]]
    wins = int(np.count_nonzero(differences > 0.0))
    ties = int(np.count_nonzero(differences == 0.0))
    losses = int(np.count_nonzero(differences < 0.0))
    usable_count = int(len(differences))
    return {
        "pair_count": total,
        "usable_pair_count": usable_count,
        "usable_pair_rate": float(usable_count / total),
        "win_count": wins,
        "tie_count": ties,
        "loss_count": losses,
        "win_rate": None if usable_count == 0 else float(wins / usable_count),
        "win_rate_wilson95_lower": wilson_lower_bound(wins, usable_count),
        "median_correct_minus_wrong": (
            None if usable_count == 0 else float(np.median(differences))
        ),
        "p10_correct_minus_wrong": (
            None if usable_count == 0 else float(np.quantile(differences, 0.1))
        ),
    }


def hard_pair_raw_screen(
    metrics: Mapping[str, object],
    *,
    minimum_usable_pairs: int,
    minimum_usable_pair_rate: float,
) -> dict[str, object]:
    """Conservative train-only gate for a raw hard-pair feature signal."""

    if int(minimum_usable_pairs) <= 0 or not 0.0 <= float(minimum_usable_pair_rate) <= 1.0:
        raise ValueError("hard-pair gate thresholds are invalid")
    usable = int(metrics["usable_pair_count"])
    rate = metrics["usable_pair_rate"]
    lower = metrics["win_rate_wilson95_lower"]
    median = metrics["median_correct_minus_wrong"]
    checks = {
        "minimum_usable_pairs": usable >= int(minimum_usable_pairs),
        "minimum_usable_pair_rate": (
            rate is not None and float(rate) >= float(minimum_usable_pair_rate)
        ),
        "wilson95_lower_above_chance": lower is not None and float(lower) > 0.5,
        "positive_median_gap": median is not None and float(median) > 0.0,
    }
    return {
        "policy": (
            "train-only raw hard-pair screen; a pass only permits a separately frozen "
            "validation export and never calibration, fusion, or pose scoring"
        ),
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def _rank_histogram(ranks: np.ndarray) -> dict[str, int]:
    values = np.asarray(ranks, dtype=np.int64).reshape(-1)
    return {str(int(rank)): int(np.count_nonzero(values == rank)) for rank in np.unique(values)}


def audit_frozen_fulltrack_hard_pairs(
    *,
    appearance_artifacts: Sequence[Path],
    colmap_model_dir: Path,
    projected_landmark_bank: Path,
    output_dir: Path,
    audit_splits: Sequence[str],
    registered_identity_radius_px: float,
    minimum_shift_pairs: int,
    minimum_shift_m: float,
    maximum_shift_dispersion_m: float,
    maximum_shift_relative_dispersion: float,
    minimum_usable_pairs: int,
    minimum_usable_pair_rate: float,
) -> dict[str, object]:
    """Join registered identities to frozen full-track features for hard-pair audit."""

    if float(registered_identity_radius_px) <= 0.0:
        raise ValueError("registered identity radius must be positive")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite hard-pair audit output")
    arrays, metadata = merge_frozen_fulltrack_appearance(tuple(appearance_artifacts))
    _validate_projected_bank_lineage(metadata, projected_landmark_bank=Path(projected_landmark_bank))
    bank, _bank_metadata = load_landmark_index_npz(Path(projected_landmark_bank))
    xyz_by_track = {
        int(track): np.asarray(bank.xyz[index], dtype=np.float64)
        for index, track in enumerate(np.asarray(bank.track_ids, dtype=np.int64))
    }
    query_ids = np.asarray(arrays["verification_query_ids"]).astype(str)
    split_names = np.asarray(arrays["split_names"]).astype(str)
    selected = np.isin(split_names, tuple(str(value) for value in audit_splits))
    if not np.any(selected):
        raise ValueError("hard-pair audit has no requested split rows")
    images = read_colmap_images_binary(Path(colmap_model_dir) / "images.bin")
    images_by_name = {str(image.image_name): image for image in images.values()}
    targets = registered_query_observation_targets(
        query_ids=query_ids,
        query_xy=np.asarray(arrays["verification_xy"], dtype=np.float32),
        images_by_name=images_by_name,
        max_distance_px=float(registered_identity_radius_px),
    )
    labels = registered_candidate_identity_labels(
        np.asarray(arrays["candidate_track_ids"], dtype=np.int64), targets
    )
    selection_all = select_rank2_to_top1_wrong_pairs(
        query_ids=query_ids,
        candidate_track_ids=np.asarray(arrays["candidate_track_ids"], dtype=np.int64),
        candidate_probabilities=np.asarray(arrays["candidate_probabilities"], dtype=np.float32),
        correct_labels=labels,
        xyz_by_track=xyz_by_track,
    )
    in_split = selected[selection_all.row_indices]
    selection = HardPairSelection(
        row_indices=selection_all.row_indices[in_split],
        positive_columns=selection_all.positive_columns[in_split],
        negative_columns=selection_all.negative_columns[in_split],
        positive_ranks=selection_all.positive_ranks[in_split],
        query_ids=selection_all.query_ids[in_split],
        wrong_minus_correct_xyz=selection_all.wrong_minus_correct_xyz[in_split],
    )
    coherent_mask, coherent_diagnostics = coherent_shift_mask(
        selection,
        minimum_pairs=int(minimum_shift_pairs),
        minimum_shift_m=float(minimum_shift_m),
        maximum_dispersion_m=float(maximum_shift_dispersion_m),
        maximum_relative_dispersion=float(maximum_shift_relative_dispersion),
    )
    feature_names = np.asarray(arrays["feature_names"]).astype(str)
    values = np.asarray(arrays["candidate_summary_features"], dtype=np.float32)
    feature_valid = np.asarray(arrays["candidate_summary_feature_valid"], dtype=bool)
    result: dict[str, object] = {
        "format": ARTIFACT_FORMAT,
        "stage": "audit_frozen_fulltrack_hard_pairs",
        "appearance_artifacts": [
            {"path": str(path), "sha256": file_sha256_short(path)}
            for path in appearance_artifacts
        ],
        "projected_landmark_bank": {
            "path": str(projected_landmark_bank),
            "sha256": file_sha256_short(projected_landmark_bank),
        },
        "audit_splits": [str(value) for value in audit_splits],
        "registered_identity_radius_px": float(registered_identity_radius_px),
        "protocol": {
            "feature_export_target_free": True,
            "targets_joined_only_after_frozen_export": True,
            "candidate_identity_and_prior_fixed": True,
            "visual_feature_not_used_to_define_hard_pairs": True,
            "calibration_or_fusion_fitted": False,
            "pose_scoring_performed": False,
            "test_used_for_model_selection": False,
        },
        "evaluation_scope": {
            "requested_split_row_count": int(np.sum(selected)),
            "hard_pair_count": selection.pair_count,
            "hard_query_count": int(len(set(selection.query_ids.tolist()))),
            "coherent_shift_pair_count": int(np.count_nonzero(coherent_mask)),
            "coherent_shift_query_count": int(
                sum(1 for item in coherent_diagnostics if bool(item["coherent"]))
            ),
            "positive_rank_histogram": _rank_histogram(selection.positive_ranks),
        },
        "coherent_shift_thresholds": {
            "minimum_pairs": int(minimum_shift_pairs),
            "minimum_shift_m": float(minimum_shift_m),
            "maximum_dispersion_m": float(maximum_shift_dispersion_m),
            "maximum_relative_dispersion": float(maximum_shift_relative_dispersion),
        },
        "coherent_shift_queries": coherent_diagnostics,
        "features": {},
    }
    rows_for_csv: list[dict[str, object]] = []
    for feature_index, feature_name in enumerate(feature_names.tolist()):
        all_metrics = pairwise_feature_metrics(
            scores=values[..., feature_index],
            feature_valid=feature_valid[..., feature_index],
            selection=selection,
        )
        coherent_metrics = pairwise_feature_metrics(
            scores=values[..., feature_index],
            feature_valid=feature_valid[..., feature_index],
            selection=selection,
            subset_mask=coherent_mask,
        )
        payload = {
            "all_rank2_to_top1_wrong": all_metrics,
            "coherent_shift_subset": coherent_metrics,
            "train_only_raw_screen": hard_pair_raw_screen(
                all_metrics,
                minimum_usable_pairs=int(minimum_usable_pairs),
                minimum_usable_pair_rate=float(minimum_usable_pair_rate),
            )
            if set(audit_splits) == {"train"}
            else None,
        }
        result["features"][str(feature_name)] = payload
        rows_for_csv.append(
            {
                "feature_name": str(feature_name),
                "all_usable_pair_count": all_metrics["usable_pair_count"],
                "all_usable_pair_rate": all_metrics["usable_pair_rate"],
                "all_win_rate": all_metrics["win_rate"],
                "all_wilson95_lower": all_metrics["win_rate_wilson95_lower"],
                "all_median_gap": all_metrics["median_correct_minus_wrong"],
                "coherent_usable_pair_count": coherent_metrics["usable_pair_count"],
                "coherent_win_rate": coherent_metrics["win_rate"],
                "coherent_wilson95_lower": coherent_metrics["win_rate_wilson95_lower"],
                "coherent_median_gap": coherent_metrics["median_correct_minus_wrong"],
                "train_screen_pass": (
                    None
                    if payload["train_only_raw_screen"] is None
                    else payload["train_only_raw_screen"]["passed"]
                ),
            }
        )
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (output / "feature_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_for_csv[0]))
        writer.writeheader()
        writer.writerows(rows_for_csv)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--appearance-artifacts", required=True)
    parser.add_argument("--colmap-model-dir", required=True)
    parser.add_argument("--projected-landmark-bank", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--audit-splits", default="train")
    parser.add_argument("--registered-identity-radius-px", type=float, default=2.0)
    parser.add_argument("--minimum-shift-pairs", type=int, default=3)
    parser.add_argument("--minimum-shift-m", type=float, default=0.10)
    parser.add_argument("--maximum-shift-dispersion-m", type=float, default=0.20)
    parser.add_argument("--maximum-shift-relative-dispersion", type=float, default=0.35)
    parser.add_argument("--minimum-usable-pairs", type=int, default=100)
    parser.add_argument("--minimum-usable-pair-rate", type=float, default=0.50)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = audit_frozen_fulltrack_hard_pairs(
        appearance_artifacts=_paths(args.appearance_artifacts),
        colmap_model_dir=Path(args.colmap_model_dir),
        projected_landmark_bank=Path(args.projected_landmark_bank),
        output_dir=Path(args.output_dir),
        audit_splits=_splits(args.audit_splits),
        registered_identity_radius_px=float(args.registered_identity_radius_px),
        minimum_shift_pairs=int(args.minimum_shift_pairs),
        minimum_shift_m=float(args.minimum_shift_m),
        maximum_shift_dispersion_m=float(args.maximum_shift_dispersion_m),
        maximum_shift_relative_dispersion=float(args.maximum_shift_relative_dispersion),
        minimum_usable_pairs=int(args.minimum_usable_pairs),
        minimum_usable_pair_rate=float(args.minimum_usable_pair_rate),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
