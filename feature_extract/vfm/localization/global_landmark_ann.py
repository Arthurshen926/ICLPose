"""Persistent full-bank FAISS IVF search without image retrieval or submaps."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex, normalize_rows


@dataclass(frozen=True)
class FaissIVFConfig:
    nlist: int = 1024
    train_samples: int = 100_000
    seed: int = 0

    def __post_init__(self) -> None:
        if int(self.nlist) <= 0 or int(self.train_samples) <= 0:
            raise ValueError("nlist and train_samples must be positive")


@dataclass(frozen=True)
class UniqueTrackSearchResult:
    bank_row_indices: np.ndarray
    track_ids: np.ndarray
    prototype_ids: np.ndarray
    scores: np.ndarray
    search_k: int
    nprobe: int

    def __post_init__(self) -> None:
        shape = np.asarray(self.bank_row_indices).shape
        if len(shape) != 2:
            raise ValueError("search result arrays must have shape (N, L)")
        for value in (self.track_ids, self.prototype_ids, self.scores):
            if np.asarray(value).shape != shape:
                raise ValueError("search result arrays must have matching shapes")


class PersistentFaissIVFIndex:
    """Strictly bound FAISS IndexIVFFlat over one descriptor-space bank."""

    def __init__(self, faiss_index: Any, metadata: dict[str, object]) -> None:
        self.index = faiss_index
        self.metadata = dict(metadata)

    @property
    def feature_dim(self) -> int:
        return int(self.index.d)

    @property
    def size(self) -> int:
        return int(self.index.ntotal)

    def search_unique_tracks(
        self,
        query_features: np.ndarray,
        landmark_index: LandmarkMapIndex,
        *,
        proposal_top_l: int,
        nprobe: int,
        oversample_factor: int = 4,
    ) -> UniqueTrackSearchResult:
        if int(proposal_top_l) <= 0 or int(nprobe) <= 0 or int(oversample_factor) <= 0:
            raise ValueError("proposal_top_l, nprobe and oversample_factor must be positive")
        if len(landmark_index) != self.size:
            raise ValueError("landmark bank row count does not match FAISS index")
        queries, valid = normalize_rows(np.asarray(query_features, dtype=np.float32))
        if not np.all(valid):
            raise ValueError("query descriptors contain zero or non-finite rows")
        _tracks, prototype_counts = np.unique(landmark_index.track_ids, return_counts=True)
        max_prototypes = int(np.max(prototype_counts)) if prototype_counts.size else 1
        search_k = min(
            self.size,
            max(
                int(proposal_top_l),
                int(proposal_top_l) * max_prototypes * int(oversample_factor),
            ),
        )
        self.index.nprobe = min(int(nprobe), int(self.index.nlist))
        raw_scores, raw_rows = self.index.search(
            np.ascontiguousarray(queries.astype(np.float32, copy=False)),
            int(search_k),
        )
        rows, tracks, prototypes, scores = collapse_unique_track_search(
            raw_rows,
            raw_scores,
            landmark_index.track_ids,
            landmark_index.prototype_ids,
            proposal_top_l=int(proposal_top_l),
        )
        return UniqueTrackSearchResult(
            bank_row_indices=rows,
            track_ids=tracks,
            prototype_ids=prototypes,
            scores=scores,
            search_k=int(search_k),
            nprobe=int(self.index.nprobe),
        )


def collapse_unique_track_search(
    raw_bank_rows: np.ndarray,
    raw_scores: np.ndarray,
    bank_track_ids: np.ndarray,
    bank_prototype_ids: np.ndarray,
    *,
    proposal_top_l: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collapse multi-prototype FAISS rows to stable unique-track top-L rows."""

    raw_rows = np.asarray(raw_bank_rows, dtype=np.int64)
    values = np.asarray(raw_scores, dtype=np.float32)
    if raw_rows.ndim != 2 or values.shape != raw_rows.shape:
        raise ValueError("raw FAISS rows and scores must have matching shape (N, K)")
    track_ids = np.asarray(bank_track_ids, dtype=np.int64).reshape(-1)
    prototype_ids = np.asarray(bank_prototype_ids, dtype=np.int64).reshape(-1)
    if track_ids.shape != prototype_ids.shape:
        raise ValueError("bank track and prototype ids must have matching shapes")
    top_l = int(proposal_top_l)
    if top_l <= 0:
        raise ValueError("proposal_top_l must be positive")
    output_rows = np.full((raw_rows.shape[0], top_l), -1, dtype=np.int64)
    output_tracks = np.full_like(output_rows, -1)
    output_prototypes = np.full_like(output_rows, -1)
    output_scores = np.full(output_rows.shape, -np.inf, dtype=np.float32)
    for query_row in range(raw_rows.shape[0]):
        seen: set[int] = set()
        output_column = 0
        order = np.argsort(-values[query_row], kind="stable")
        for raw_column in order.tolist():
            bank_row = int(raw_rows[query_row, raw_column])
            if bank_row < 0 or bank_row >= len(track_ids):
                continue
            track_id = int(track_ids[bank_row])
            if track_id in seen:
                continue
            seen.add(track_id)
            output_rows[query_row, output_column] = bank_row
            output_tracks[query_row, output_column] = track_id
            output_prototypes[query_row, output_column] = int(prototype_ids[bank_row])
            output_scores[query_row, output_column] = float(values[query_row, raw_column])
            output_column += 1
            if output_column >= top_l:
                break
    return output_rows, output_tracks, output_prototypes, output_scores


def _expected_metadata(
    *,
    landmark_bank_path: Path,
    landmark_index: LandmarkMapIndex,
    descriptor_space_id: str,
    config: FaissIVFConfig,
) -> dict[str, object]:
    return {
        "format": "full_bank_faiss_ivf_flat_v1",
        "landmark_bank": str(Path(landmark_bank_path)),
        "landmark_bank_sha256": file_sha256_short(Path(landmark_bank_path)),
        "descriptor_space_id": str(descriptor_space_id),
        "feature_count": int(len(landmark_index)),
        "feature_dim": int(landmark_index.feature_dim),
        "config": asdict(config),
        "normalization": "row_l2",
        "metric": "inner_product",
        "row_mapping": "identity_landmark_bank_row",
    }


def build_or_load_faiss_ivf_index(
    landmark_index: LandmarkMapIndex,
    *,
    landmark_bank_path: Path,
    descriptor_space_id: str,
    cache_path: Path,
    config: FaissIVFConfig,
) -> tuple[PersistentFaissIVFIndex, bool]:
    """Load a matching cache or build it once; stale caches are rejected."""

    try:
        import faiss
    except Exception as exc:  # pragma: no cover - depends on deployment environment.
        raise RuntimeError("FAISS is required for full-bank IVF search") from exc
    output = Path(cache_path)
    manifest_path = output.with_suffix(output.suffix + ".json")
    expected = _expected_metadata(
        landmark_bank_path=Path(landmark_bank_path),
        landmark_index=landmark_index,
        descriptor_space_id=str(descriptor_space_id),
        config=config,
    )
    if output.exists() != manifest_path.exists():
        raise ValueError("FAISS cache and manifest must either both exist or both be absent")
    if output.exists():
        actual = json.loads(manifest_path.read_text())
        mismatches = {
            key: {"expected": value, "actual": actual.get(key)}
            for key, value in expected.items()
            if actual.get(key) != value
        }
        if mismatches:
            raise ValueError(f"stale FAISS cache metadata: {json.dumps(mismatches, sort_keys=True)}")
        index = faiss.read_index(str(output))
        if int(index.ntotal) != len(landmark_index) or int(index.d) != int(landmark_index.feature_dim):
            raise ValueError("FAISS cache dimensions do not match landmark bank")
        return PersistentFaissIVFIndex(index, actual), True

    features, valid = normalize_rows(np.asarray(landmark_index.features, dtype=np.float32))
    if not np.all(valid):
        raise ValueError("landmark bank contains invalid descriptors")
    effective_nlist = min(int(config.nlist), max(1, int(len(features))))
    quantizer = faiss.IndexFlatIP(int(features.shape[1]))
    index = faiss.IndexIVFFlat(
        quantizer,
        int(features.shape[1]),
        int(effective_nlist),
        faiss.METRIC_INNER_PRODUCT,
    )
    rng = np.random.default_rng(int(config.seed))
    train_count = min(int(config.train_samples), int(len(features)))
    train_rows = rng.choice(len(features), size=train_count, replace=False)
    train_rows.sort()
    index.train(np.ascontiguousarray(features[train_rows]))
    index.add(np.ascontiguousarray(features))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    faiss.write_index(index, str(temporary))
    os.replace(temporary, output)
    metadata = {
        **expected,
        "effective_nlist": int(effective_nlist),
        "faiss_version": str(getattr(faiss, "__version__", "unknown")),
        "trained_sample_count": int(train_count),
    }
    manifest_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return PersistentFaissIVFIndex(index, metadata), False


def audit_ann_against_exact_candidates(
    ann_track_ids: np.ndarray,
    exact_track_ids: np.ndarray,
    correct_track_ids: np.ndarray,
) -> dict[str, float | int]:
    """Measure whether an approximate global index loses exact top-L positives."""

    ann = np.asarray(ann_track_ids, dtype=np.int64)
    exact = np.asarray(exact_track_ids, dtype=np.int64)
    correct = np.asarray(correct_track_ids, dtype=np.int64).reshape(-1)
    if ann.shape != exact.shape or ann.ndim != 2 or ann.shape[0] != correct.shape[0]:
        raise ValueError("ANN, exact and correct track arrays have incompatible shapes")
    ann_present = np.any(ann == correct[:, None], axis=1)
    exact_present = np.any(exact == correct[:, None], axis=1)
    overlaps = []
    for ann_row, exact_row in zip(ann, exact):
        ann_set = {int(value) for value in ann_row if int(value) >= 0}
        exact_set = {int(value) for value in exact_row if int(value) >= 0}
        overlaps.append(len(ann_set & exact_set) / max(len(exact_set), 1))
    return {
        "sample_count": int(len(correct)),
        "exact_correct_track_recall": float(np.mean(exact_present)),
        "ann_correct_track_recall": float(np.mean(ann_present)),
        "correct_track_retention_given_exact": (
            0.0 if not np.any(exact_present) else float(np.mean(ann_present[exact_present]))
        ),
        "top1_track_agreement": float(np.mean(ann[:, 0] == exact[:, 0])),
        "mean_exact_top_l_track_overlap": float(np.mean(overlaps)),
        "ann_full_row_rate": float(np.mean(np.all(ann >= 0, axis=1))),
    }
