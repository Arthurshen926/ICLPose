import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.visualize_patch_to_3d_vfm_matches import main
from feature_extract.vfm.map_lifting import SelectedTrackFeatureBank, TrackFeature, save_selected_track_bank_npz
from feature_extract.vfm.query_to_3d_matching import token_grid_xy
from feature_extract.vfm.tokens import TokenBankManifest, TokenBankRecord, TokenLayerSpec


def test_visualize_patch_to_3d_cli_writes_patch_outputs(tmp_path: Path) -> None:
    dim = 4
    query_map = np.zeros((dim, 3, 3), dtype=np.float32)
    query_map[:, 1, 1] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
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
    center = token_grid_xy(3, 3, image_width=100, image_height=100)[4]
    xyz = np.asarray([(center[0] - 50.0) / 80.0 * 5.0, (center[1] - 50.0) / 80.0 * 5.0, 5.0])
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
                    ("ref.png",),
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
                "image_id": "ref.png",
                "point2d_idx": 0,
                "xy": center.tolist(),
                "xyz": xyz.tolist(),
                "track_length": 3,
                "reprojection_error": 0.1,
            }
        )
        + "\n"
    )
    pose_path = tmp_path / "poses.txt"
    pose_path.write_text(
        "Visual Landmark Dataset V1\n"
        "ImageFile, Camera Position [X Y Z W P Q R]\n\n"
        "q.png 0 0 0 1 0 0 0\n"
    )
    try:
        import cv2
    except Exception:
        return
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, :, 1] = 60
    cv2.imwrite(str(tmp_path / "q.png"), image)
    output_dir = tmp_path / "viz"

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
            "--image_root",
            str(tmp_path),
            "--query_id",
            "q.png",
            "--submap_mode",
            "none",
            "--default_camera",
            "1,100,100,80,80,50,50",
            "--match_mode",
            "nn",
            "--top_k",
            "1",
            "--min_similarity",
            "0.5",
            "--output_dir",
            str(output_dir),
        ]
    )

    summary = json.loads((output_dir / "visualization_summary.json").read_text())
    assert summary["stage"] == "patch_to_3d_vfm_match_visualization"
    item = summary["queries"][0]
    assert Path(item["overlay_png"]).exists()
    assert Path(item["projection_rgb_png"]).exists()
    assert item["overlay"]["patch_precision"] == 1.0
