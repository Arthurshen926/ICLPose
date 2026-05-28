import json
import struct
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.eval_query_to_3d_vfm_matching import main
from feature_extract.vfm.colmap_tracks import ColmapTrackObservation
from feature_extract.vfm.landmark_visibility import LandmarkVisibilityIndex
from feature_extract.vfm.map_lifting import SelectedTrackFeatureBank, TrackFeature, save_selected_track_bank_npz
from feature_extract.vfm.query_to_3d_matching import token_grid_xy
from feature_extract.vfm.tokens import TokenBankManifest, TokenBankRecord, TokenLayerSpec


def _write_simple_cameras_bin(path: Path, *, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<iiQQ", 7, 1, width, height))
        handle.write(struct.pack("<dddd", 80.0, 80.0, width / 2.0, height / 2.0))


def test_eval_query_to_3d_cli_writes_pose_metrics(tmp_path: Path) -> None:
    dim = 8
    height = 4
    width = 4
    grid_xy = token_grid_xy(width, height, image_width=100, image_height=100)
    token_indices = np.asarray([0, 3, 5, 6, 9, 12], dtype=np.int64)
    z = np.asarray([4.0, 4.6, 5.2, 5.8, 6.4, 7.0], dtype=np.float64)
    xy = grid_xy[token_indices]
    xyz = np.stack([(xy[:, 0] - 50.0) / 80.0 * z, (xy[:, 1] - 50.0) / 80.0 * z, z], axis=1)

    query_map = np.zeros((dim, height, width), dtype=np.float32)
    tracks = {}
    track_lines = []
    for local_idx, token_idx in enumerate(token_indices):
        track_id = 10 + local_idx
        feature = np.zeros((dim,), dtype=np.float32)
        feature[local_idx] = 1.0
        y_idx, x_idx = divmod(int(token_idx), width)
        query_map[:, y_idx, x_idx] = feature
        tracks[track_id] = TrackFeature(
            track_id,
            feature,
            np.zeros((dim,), dtype=np.float32),
            2,
            1.0,
            ("ref_a.png",),
        )
        track_lines.append(
            json.dumps(
                {
                    "track_id": track_id,
                    "image_id": "ref_a.png",
                    "point2d_idx": local_idx,
                    "xy": xy[local_idx].tolist(),
                    "xyz": xyz[local_idx].tolist(),
                    "track_length": 2,
                    "reprojection_error": 0.1,
                    "camera_id": 1,
                    "image_width": 100,
                    "image_height": 100,
                }
            )
        )

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
    bank_path = tmp_path / "bank.npz"
    save_selected_track_bank_npz(SelectedTrackFeatureBank(tracks=tracks, feature_dim=dim), bank_path)
    tracks_path = tmp_path / "tracks.jsonl"
    tracks_path.write_text("\n".join(track_lines) + "\n")
    visibility_path = tmp_path / "visibility.npz"
    LandmarkVisibilityIndex.from_observations(
        [
            ColmapTrackObservation(
                track_id=int(10 + idx),
                image_id="ref_a.png",
                point2d_idx=idx,
                xy=(0.0, 0.0),
                xyz=xyz[idx],
                track_length=2,
                reprojection_error=0.1,
            )
            for idx in range(len(token_indices))
        ]
        + [
            ColmapTrackObservation(
                track_id=999,
                image_id="ref_a.png",
                point2d_idx=99,
                xy=(0.0, 0.0),
                xyz=np.asarray([0.0, 0.0, 5.0], dtype=np.float64),
                track_length=2,
                reprojection_error=0.1,
            )
        ]
    ).save_npz(visibility_path)
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
                "reference_image": "ref_a.png",
                "metadata": {"retrieval_rank": 1},
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
            "--landmark_bank",
            str(bank_path),
            "--track_observations",
            str(tracks_path),
            "--visibility_index",
            str(visibility_path),
            "--query_pose_file",
            str(pose_path),
            "--candidate_bank",
            str(candidate_path),
            "--submap_mode",
            "reference_visibility",
            "--default_camera",
            "1,100,100,80,80,50,50",
            "--query_token_step",
            "1",
            "--min_similarity",
            "0.5",
            "--ratio_threshold",
            "0.8",
            "--mutual",
            "--output_jsonl",
            str(rows_path),
            "--summary_json",
            str(summary_path),
        ]
    )

    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    summary = json.loads(summary_path.read_text())
    assert rows[0]["pnp_success"] is True
    assert rows[0]["full_visible_tracks"] == 7
    assert rows[0]["bank_visible_tracks"] == 6
    assert rows[0]["projected_landmarks"] == 6
    assert rows[0]["pnp_inlier_count"] >= 5
    assert rows[0]["translation_error_m"] < 1e-4
    assert summary["query_count"] == 1
    assert summary["success_25cm_10deg"] == 1.0


def test_eval_query_to_3d_cli_auto_loads_colmap_camera_from_pose_scene_dir(tmp_path: Path) -> None:
    dim = 4
    query_map = np.zeros((dim, 1, 1), dtype=np.float32)
    query_map[0, 0, 0] = 1.0
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

    bank_path = tmp_path / "bank.npz"
    save_selected_track_bank_npz(
        SelectedTrackFeatureBank(
            tracks={
                1: TrackFeature(
                    1,
                    np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    np.zeros((dim,), dtype=np.float32),
                    3,
                    1.0,
                    ("ref_a.png",),
                )
            },
            feature_dim=dim,
        ),
        bank_path,
    )
    tracks_path = tmp_path / "tracks.jsonl"
    tracks_path.write_text(
        json.dumps(
            {
                "track_id": 1,
                "image_id": "ref_a.png",
                "point2d_idx": 0,
                "xy": [50.0, 50.0],
                "xyz": [0.0, 0.0, 5.0],
                "track_length": 3,
                "reprojection_error": 0.1,
                "camera_id": 7,
                "image_width": 100,
                "image_height": 100,
            }
        )
        + "\n"
    )
    pose_path = tmp_path / "dataset_test.txt"
    pose_path.write_text(
        "Visual Landmark Dataset V1\n"
        "ImageFile, Camera Position [X Y Z W P Q R]\n\n"
        "q.png 0 0 0 1 0 0 0\n"
    )
    _write_simple_cameras_bin(tmp_path / "sparse" / "0" / "cameras.bin", width=100, height=100)
    rows_path = tmp_path / "rows.jsonl"
    summary_path = tmp_path / "summary.json"

    main(
        [
            "--query_manifest",
            str(manifest_path),
            "--landmark_bank",
            str(bank_path),
            "--track_observations",
            str(tracks_path),
            "--query_pose_file",
            str(pose_path),
            "--submap_mode",
            "none",
            "--default_camera",
            "1,10,10,8,8,5,5",
            "--query_token_step",
            "1",
            "--min_similarity",
            "0.5",
            "--min_observation_count",
            "1",
            "--output_jsonl",
            str(rows_path),
            "--summary_json",
            str(summary_path),
        ]
    )

    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    summary = json.loads(summary_path.read_text())
    assert rows[0]["coordinate_audit"]["camera_width"] == 100
    assert rows[0]["coordinate_audit"]["camera_height"] == 100
    assert summary["camera"]["source"].endswith("sparse/0/cameras.bin")
