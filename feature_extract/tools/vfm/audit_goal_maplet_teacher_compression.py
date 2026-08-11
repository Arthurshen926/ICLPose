"""Audit mapping-only teacher compression before physical-readout training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.train_goal_maplet_physical_instance_readout import _compress
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


TEACHERS = {
    "dino_v3_7b": 17011,
    "sam3": 17013,
    "siglip2-g": 17017,
}


def _normalized(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1.0e-8)


def _metrics(reference: np.ndarray, candidate: np.ndarray, topk: int) -> dict[str, float]:
    count = int(reference.shape[0])
    off_diagonal = ~np.eye(count, dtype=bool)
    reference_similarity = reference @ reference.T
    candidate_similarity = candidate @ candidate.T
    correlation = float(np.corrcoef(
        reference_similarity[off_diagonal], candidate_similarity[off_diagonal],
    )[0, 1])
    reference_rank = np.argsort(
        -np.where(off_diagonal, reference_similarity, -np.inf), axis=1,
    )[:, : int(topk)]
    candidate_rank = np.argsort(
        -np.where(off_diagonal, candidate_similarity, -np.inf), axis=1,
    )[:, : int(topk)]
    overlap = np.mean([
        len(set(left).intersection(right)) / max(float(topk), 1.0)
        for left, right in zip(reference_rank, candidate_rank)
    ])
    return {"pairwise_cosine_correlation": correlation, "topk_neighbour_retention": float(overlap)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher_cache", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--maximum_files", type=int, default=32)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError(f"refusing to overwrite {output}")
    paths = sorted(Path(args.teacher_cache).glob("*.npz"))[: int(args.maximum_files)]
    if not paths:
        raise ValueError("teacher compression audit found no teacher cache")
    variants = {
        "legacy_block_mean64": ("legacy_block_mean", 64),
        "signed_block_sketch64": ("signed_block_sketch", 64),
        "signed_block_sketch128": ("signed_block_sketch", 128),
    }
    report = {}
    for teacher, seed in TEACHERS.items():
        rows = {name: [] for name in variants}
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                reference = _normalized(np.asarray(data[teacher], dtype=np.float32))
            for name, (method, dimensions) in variants.items():
                candidate = _compress(
                    reference,
                    dimensions=int(dimensions),
                    method=str(method),
                    seed=int(seed),
                )
                rows[name].append(_metrics(reference, candidate, int(args.topk)))
        report[teacher] = {
            name: {
                key: float(np.mean([row[key] for row in values]))
                for key in values[0]
            }
            for name, values in rows.items()
        }
    result = {
        "stage": "goal_maplet_mapping_teacher_compression_audit",
        "teacher_cache": str(args.teacher_cache),
        "sample_file_count": len(paths),
        "sample_file_sha256": [file_sha256(path) for path in paths],
        "topk": int(args.topk),
        "teachers": report,
        "deployment_contract": {
            "mapping_only": True,
            "teacher_compression_stored_at_deployment": False,
            "stored_downstream_embedding_count": 0,
            "stores_mapping_rgb": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
