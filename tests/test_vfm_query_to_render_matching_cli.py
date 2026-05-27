import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.eval_query_to_render_vfm_matching import main
from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.gaussian_vfm_field import GaussianVFMField
from feature_extract.vfm.query_to_3d_matching import token_grid_xy
from feature_extract.vfm.tokens import TokenBankManifest, TokenBankRecord, TokenLayerSpec


def _xyz_from_xy(xy: np.ndarray, z: np.ndarray) -> np.ndarray:
    x = (xy[:, 0] - 50.0) / 80.0 * z
    y = (xy[:, 1] - 50.0) / 80.0 * z
    return np.stack([x, y, z], axis=1).astype(np.float64)


def test_eval_query_to_render_cli_writes_pose_metrics(tmp_path: Path) -> None:
    dim = 8
    height = 4
    width = 4
    token_indices = np.asarray([0, 3, 5, 6, 9, 12], dtype=np.int64)
    xy = token_grid_xy(width, height, image_width=100, image_height=100)[token_indices]
    xyz = _xyz_from_xy(xy, np.asarray([4.0, 4.6, 5.2, 5.8, 6.4, 7.0], dtype=np.float64))

    query_map = np.zeros((dim, height, width), dtype=np.float32)
    features = []
    for local_idx, token_idx in enumerate(token_indices):
        feature = np.zeros((dim,), dtype=np.float32)
        feature[local_idx] = 1.0
        y_idx, x_idx = divmod(int(token_idx), width)
        query_map[:, y_idx, x_idx] = feature
        features.append(feature)

    token_path = tmp_path / "q.npz"
    np.savez_compressed(token_path, radio_final=query_map)
    manifest = TokenBankManifest(
        records=(
            TokenBankRecord(
                image_id="q.png",
                token_path=token_path,
                layers=(TokenLayerSpec("radio_final", "synthetic", "final", dim, 16),),
                split="test",
                scene="Synthetic",
            ),
        )
    )
    manifest_path = tmp_path / "manifest.json"
    manifest.to_json(manifest_path)

    field_path = tmp_path / "field.npz"
    GaussianVFMField(
        xyz=xyz,
        features=np.stack(features, axis=0),
        opacity=np.ones((len(token_indices),), dtype=np.float32),
        scale=np.ones((len(token_indices),), dtype=np.float32) * 0.05,
        gaussian_indices=np.arange(len(token_indices), dtype=np.int64),
        nearest_track_ids=np.arange(len(token_indices), dtype=np.int64),
        support_counts=np.ones((len(token_indices),), dtype=np.int64),
        mean_distances=np.zeros((len(token_indices),), dtype=np.float32),
        metadata={"source": "synthetic"},
    ).save_npz(field_path)

    pose_path = tmp_path / "poses.txt"
    pose_path.write_text(
        "Visual Landmark Dataset V1\n"
        "ImageFile, Camera Position [X Y Z W P Q R]\n\n"
        "q.png 0 0 0 1 0 0 0\n"
    )
    candidate_path = tmp_path / "candidates.jsonl"
    candidate_path.write_text(
        json.dumps({"record_type": "header", "protocol_name": "synthetic"}) + "\n"
        + json.dumps(
            {
                "record_type": "candidate",
                "candidate_id": "q:0",
                "query_id": "q.png",
                "reference_image": "ref.png",
                "metadata": {"retrieval_rank": 1},
                "pose": np.eye(4, dtype=np.float64).tolist(),
            }
        )
        + "\n"
    )
    rows_path = tmp_path / "rows.jsonl"
    summary_path = tmp_path / "summary.json"

    main(
        [
            "--query_manifest",
            str(manifest_path),
            "--field",
            str(field_path),
            "--query_pose_file",
            str(pose_path),
            "--candidate_bank",
            str(candidate_path),
            "--default_camera",
            "1,100,100,80,80,50,50",
            "--render_width",
            "100",
            "--render_height",
            "100",
            "--render_radius_px",
            "1.0",
            "--query_token_step",
            "1",
            "--min_similarity",
            "0.5",
            "--ratio_threshold",
            "0.8",
            "--mutual",
            "--candidate_top_n",
            "1",
            "--output_jsonl",
            str(rows_path),
            "--summary_json",
            str(summary_path),
        ]
    )

    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    summary = json.loads(summary_path.read_text())
    assert rows[0]["pnp_success"] is True
    assert rows[0]["selected_candidate_rank"] == 1
    assert rows[0]["render_visible_pixel_count"] >= 6
    assert rows[0]["pnp_inlier_count"] >= 5
    assert rows[0]["translation_error_m"] < 1e-4
    assert summary["query_count"] == 1
    assert summary["success_25cm_10deg"] == 1.0
