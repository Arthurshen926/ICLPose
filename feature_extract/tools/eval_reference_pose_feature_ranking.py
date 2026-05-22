#!/usr/bin/env python3
"""Evaluate feature-descriptor reranking on reference-pose candidate banks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.radio_loc_dataset import read_colmap_images  # noqa: E402
from feature_extract.localizability.metrics import ranking_metrics  # noqa: E402
from feature_extract.localizability.reference_pose_scoring import (  # noqa: E402
    load_descriptor_bank,
    load_patch_descriptor_bank,
    load_descriptors_for_names,
    project_descriptor_bank_pca,
    retrieval_order_scores,
    score_reference_pose_patch_descriptors,
    score_reference_pose_descriptors,
)


def _name_to_image_id(colmap_dir: str | Path) -> dict[str, int]:
    images = read_colmap_images(str(Path(colmap_dir) / "images.bin"))
    return {meta.name: int(img_id) for img_id, meta in images.items()}


def _load_bank(path: str | Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {
        "sample_names": [str(v) for v in data["sample_names"].tolist()],
        "reference_names": [[str(x) for x in row.tolist()] for row in data["reference_names"]],
        "pose_cost_m": torch.as_tensor(data["pose_cost_m"], dtype=torch.float32),
        "trans_err_m": torch.as_tensor(data["trans_err_m"], dtype=torch.float32),
        "rot_err_deg": torch.as_tensor(data["rot_err_deg"], dtype=torch.float32),
        "valid_mask": torch.as_tensor(data["valid_mask"], dtype=torch.bool),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--colmap-dir", required=True)
    parser.add_argument("--feature-root", default=None)
    parser.add_argument("--feature-subdir", default="fine_geo")
    parser.add_argument("--descriptor-bank", default=None)
    parser.add_argument("--score-mode", choices=("feature", "patch_feature", "retrieval_order"), default="feature")
    parser.add_argument("--patch-topk", type=int, default=8)
    parser.add_argument("--pca-out-dim", type=int, default=0)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--trans-basin-m", type=float, default=1.0)
    parser.add_argument("--rot-basin-deg", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bank = _load_bank(args.bank)
    if args.score_mode == "retrieval_order":
        scores = retrieval_order_scores(bank["valid_mask"])
        feature_valid = bank["valid_mask"].clone()
        valid = bank["valid_mask"]
        descriptor_coverage = 1.0
    elif args.score_mode == "patch_feature":
        if not args.descriptor_bank:
            raise ValueError("--descriptor-bank is required when --score-mode=patch_feature")
        patch_descriptors, descriptor_metadata = load_patch_descriptor_bank(args.descriptor_bank)
        scores, feature_valid = score_reference_pose_patch_descriptors(
            sample_names=bank["sample_names"],
            reference_names=bank["reference_names"],
            patch_descriptors=patch_descriptors,
            topk=int(args.patch_topk),
        )
        valid = bank["valid_mask"] & feature_valid
        descriptor_coverage = float(feature_valid.any(dim=1).float().mean().item())
    else:
        if args.descriptor_bank:
            descriptors, descriptor_metadata = load_descriptor_bank(args.descriptor_bank)
        else:
            if not args.feature_root:
                raise ValueError("--feature-root or --descriptor-bank is required when --score-mode=feature")
            all_names = list(bank["sample_names"])
            for refs in bank["reference_names"]:
                all_names.extend(refs)
            descriptors = load_descriptors_for_names(
                names=all_names,
                name_to_image_id=_name_to_image_id(args.colmap_dir),
                feature_root=args.feature_root,
                subdir=args.feature_subdir,
            )
            descriptor_metadata = {}
        projection_metadata = None
        if int(args.pca_out_dim) > 0:
            descriptors, projection_metadata = project_descriptor_bank_pca(
                descriptors,
                out_dim=int(args.pca_out_dim),
            )
        scores, feature_valid = score_reference_pose_descriptors(
            sample_names=bank["sample_names"],
            reference_names=bank["reference_names"],
            descriptors=descriptors,
        )
        valid = bank["valid_mask"] & feature_valid
        descriptor_coverage = float(feature_valid.any(dim=1).float().mean().item())
        if projection_metadata is not None:
            descriptor_metadata = dict(descriptor_metadata)
            descriptor_metadata["projection"] = projection_metadata
    basin = (bank["trans_err_m"] <= float(args.trans_basin_m)) & (bank["rot_err_deg"] <= float(args.rot_basin_deg))
    metrics = ranking_metrics(scores, bank["pose_cost_m"], valid_mask=valid, basin_label=basin, topk=(1, 5))
    summary = {
        "bank": str(args.bank),
        "score_mode": str(args.score_mode),
        "feature_root": str(args.feature_root) if args.feature_root else None,
        "descriptor_bank": str(args.descriptor_bank) if args.descriptor_bank else None,
        "descriptor_metadata": descriptor_metadata if args.score_mode in {"feature", "patch_feature"} else {},
        "feature_subdir": str(args.feature_subdir),
        "num_queries": len(bank["sample_names"]),
        "descriptor_coverage": descriptor_coverage,
        "patch_topk": int(args.patch_topk) if args.score_mode == "patch_feature" else None,
        "pca_out_dim": int(args.pca_out_dim) if int(args.pca_out_dim) > 0 else None,
        "metrics": {key: float(value.detach().cpu()) for key, value in metrics.items()},
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
