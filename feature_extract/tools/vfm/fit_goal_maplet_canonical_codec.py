"""Fit a task-neutral canonical RADIO-final PCA codec on mapping views only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from feature_extract.vfm.localization_goal_maplet.canonical_codec import CanonicalRadioCodec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--output_codec", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--dimension", type=int, default=256)
    parser.add_argument("--samples_per_image", type=int, default=512)
    parser.add_argument("--exclude_trajectories", nargs="*", default=["seq11", "seq3", "seq5", "seq13"])
    parser.add_argument("--seed", type=int, default=194917)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output, summary = Path(args.output_codec), Path(args.summary_json)
    if not args.force and (output.exists() or summary.exists()):
        raise FileExistsError("refusing to overwrite canonical codec")
    rng = np.random.default_rng(int(args.seed))
    excluded = set(str(value) for value in args.exclude_trajectories)
    samples, image_ids, trajectories = [], [], set()
    for path in sorted(Path(args.contributors).glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        trajectory = str(metadata["trajectory_id"])
        if trajectory in excluded:
            continue
        with np.load(Path(str(metadata["token_path"])), allow_pickle=False) as data:
            raw = np.asarray(data["radio_final"], dtype=np.float32)
        rows = raw.transpose(1, 2, 0).reshape(-1, raw.shape[0])
        take = min(int(args.samples_per_image), rows.shape[0])
        selected = rng.choice(rows.shape[0], size=take, replace=False)
        value = rows[selected]
        value /= np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-8)
        samples.append(value)
        image_ids.append(str(metadata["image_id"]))
        trajectories.add(trajectory)
    matrix = np.concatenate(samples, axis=0)
    if matrix.shape[0] <= int(args.dimension):
        raise ValueError("not enough mapping-only samples for canonical PCA")
    pca = PCA(n_components=int(args.dimension), svd_solver="randomized", random_state=int(args.seed))
    pca.fit(matrix)
    codec = CanonicalRadioCodec(
        pca.mean_.astype(np.float32),
        pca.components_.astype(np.float32),
        metadata={
            "artifact_type": "goal_maplet_canonical_radio_codec_v1",
            "fit_scope": "mapping_trajectories_only",
            "fit_image_count": len(image_ids),
            "fit_trajectory_ids": sorted(trajectories),
            "input_feature": "radio_final",
            "transform": "token_l2_then_pca_then_l2",
            "task_supervision": "none",
            "stores_mapping_rgb": False,
            "stores_mapping_image_paths": False,
            "stores_mapping_image_ids": False,
            "stored_downstream_embedding_count": 0,
            "uses_radio_intermediate": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
            "seed": int(args.seed),
        },
    )
    codec.save_npz(output)
    report = {
        "stage": "fit_goal_maplet_canonical_radio_codec",
        "codec_sha256": codec.content_sha256,
        "fit_image_count": len(image_ids),
        "fit_trajectory_ids": sorted(trajectories),
        "sample_count": int(matrix.shape[0]),
        "input_dimension": int(matrix.shape[1]),
        "output_dimension": int(codec.output_dim),
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "output_codec": str(output),
    }
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
