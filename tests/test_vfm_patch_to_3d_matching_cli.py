import json
from pathlib import Path

import numpy as np
import pytest

from feature_extract.tools.vfm.eval_patch_to_3d_vfm_matching import _fixed_pose_refinement_weights
from feature_extract.tools.vfm.eval_patch_to_3d_vfm_matching import main
from feature_extract.vfm.map_lifting import SelectedTrackFeatureBank, TrackFeature, save_selected_track_bank_npz
from feature_extract.vfm.query_to_3d_matching import QueryTo3DMatch, token_grid_xy
from feature_extract.vfm.tokens import TokenBankManifest, TokenBankRecord, TokenLayerSpec


def test_eval_patch_to_3d_cli_reports_patch_metrics(tmp_path: Path) -> None:
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
    projected = center + np.asarray([12.0, 0.0], dtype=np.float64)
    z = 5.0
    xyz = np.asarray([(projected[0] - 50.0) / 80.0 * z, (projected[1] - 50.0) / 80.0 * z, z])
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
                "xy": projected.tolist(),
                "xyz": xyz.tolist(),
                "track_length": 3,
                "reprojection_error": 0.1,
                "camera_id": 1,
                "image_width": 100,
                "image_height": 100,
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
    candidate_path = tmp_path / "candidates.jsonl"
    candidate_path.write_text(
        json.dumps({"record_type": "header", "protocol_name": "synthetic"}) + "\n"
        + json.dumps(
            {
                "record_type": "candidate",
                "candidate_id": "q:0",
                "query_id": "q.png",
                "reference_image": "ref_a.png",
                "metadata": {"retrieval_rank": 1, "pose_cost_m": 2.0},
                "pose_error": {"translation_m": 2.0, "rotation_deg": 20.0},
            }
        )
        + "\n"
        + json.dumps(
            {
                "record_type": "candidate",
                "candidate_id": "q:1",
                "query_id": "q.png",
                "reference_image": "ref.png",
                "metadata": {"retrieval_rank": 2, "pose_cost_m": 0.2},
                "pose_error": {"translation_m": 0.2, "rotation_deg": 5.0},
            }
        )
        + "\n"
    )
    rows_path = tmp_path / "rows.jsonl"
    matches_path = tmp_path / "matches.jsonl"
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
            "--candidate_bank",
            str(candidate_path),
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
            "--enable_landmark_quality",
            "--quality_track_weight",
            "1.0",
            "--quality_variance_weight",
            "0.0",
            "--quality_reprojection_weight",
            "0.0",
            "--quality_ambiguity_weight",
            "0.0",
            "--pnp_threshold_stride_multiplier",
            "1.5",
            "--patch_offset_mode",
            "fixed_same_inlier",
            "--fixed_pose_refine_method",
            "robust_lm",
            "--robust_fixed_pose_weight_mode",
            "uniform",
            "--output_jsonl",
            str(rows_path),
            "--output_matches_jsonl",
            str(matches_path),
            "--summary_json",
            str(summary_path),
        ]
    )

    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    matches = [json.loads(line) for line in matches_path.read_text().splitlines()]
    summary = json.loads(summary_path.read_text())
    assert rows[0]["patch_geometry"]["patch_at_1"] == 1.0
    assert rows[0]["patch_geometry"]["gt_precision_5px"] == 0.0
    assert rows[0]["patch_geometry"]["gt_precision_stride"] == 1.0
    assert summary["mean_patch_at_1"] == 1.0
    assert summary["mean_gt_precision_5px"] == 0.0
    assert summary["mean_gt_precision_stride"] == 1.0
    assert summary["pnp_solve_rate"] == 0.0
    assert summary["matching_config"]["landmark_quality"]["enabled"] is True
    assert summary["matching_config"]["landmark_quality"]["track_weight"] == 1.0
    assert summary["matching_config"]["patch_offset_refinement"]["mode"] == "fixed_same_inlier"
    assert summary["matching_config"]["patch_offset_refinement"]["fixed_pose_refine"]["method"] == "robust_lm"
    assert summary["matching_config"]["patch_offset_refinement"]["fixed_pose_refine"]["weight_mode"] == "uniform"
    assert summary["mean_pnp_inlier_count"] == 0.0
    assert summary["mean_pnp_inlier_ratio"] == 0.0
    assert summary["elapsed_sec"] >= 0.0
    assert summary["success_10cm_5deg"] == 0.0
    assert summary["success_50cm_10deg"] == 0.0
    assert summary["success_1m_10deg"] == 0.0
    assert rows[0]["positive_set_stats"]["positive_landmark_count"] == 1
    assert rows[0]["positive_set_stats"]["zero_positive_token_ratio"] > 0.0
    assert rows[0]["gt_visible_bank_tracks"] == 1
    assert rows[0]["submap_gt_visible_tracks"] == 1
    assert rows[0]["visible_landmark_recall"] == 1.0
    assert rows[0]["reference_prior"]["top1_translation_error_m"] == 2.0
    assert rows[0]["reference_prior"]["oracle_translation_error_m"] == 0.2
    assert summary["visible_landmark_recall"]["mean"] == 1.0
    assert summary["visible_landmark_recall"]["median"] == 1.0
    assert summary["reference_prior"]["top1"]["median_translation_error_m"] == 2.0
    assert summary["reference_prior"]["oracle"]["median_translation_error_m"] == 0.2
    assert summary["reference_prior"]["oracle"]["success_25cm_10deg"] == 1.0
    assert matches[0]["gt_reproj_error_px"] == 12.0
    assert matches[0]["gt_reproj_error_stride"] < 1.0
    assert matches[0]["strong_positive_label"] is True
    assert matches[0]["stride_positive_label"] is True
    assert matches[0]["hard_negative_label"] is False
    assert matches[0]["baseline_reproj_residual_px"] is None
    assert matches[0]["xyz"] == xyz.tolist()


def test_offset_confidence_fixed_pose_weights_downweight_uncertain_offsets() -> None:
    confident = QueryTo3DMatch(
        token_index=0,
        xy=np.asarray([10.0, 10.0], dtype=np.float64),
        track_id=1,
        xyz=np.asarray([0.0, 0.0, 4.0], dtype=np.float64),
        similarity=0.9,
        ratio=0.0,
        landmark_variance=0.0,
        patch_offset_confidence=0.95,
        patch_offset_sigma=0.25,
        patch_offset_applied=True,
    )
    uncertain = QueryTo3DMatch(
        token_index=1,
        xy=np.asarray([20.0, 20.0], dtype=np.float64),
        track_id=2,
        xyz=np.asarray([0.2, 0.2, 5.0], dtype=np.float64),
        similarity=0.9,
        ratio=0.0,
        landmark_variance=0.0,
        patch_offset_confidence=0.15,
        patch_offset_sigma=2.0,
        patch_offset_applied=False,
    )

    weights = _fixed_pose_refinement_weights([confident, uncertain], "offset_confidence")

    assert weights is not None
    assert weights[0] > weights[1]
    assert weights[1] >= 0.05


def test_offset_consistency_fixed_pose_weights_downweight_regressed_offsets() -> None:
    improved = QueryTo3DMatch(
        token_index=0,
        xy=np.asarray([10.0, 10.0], dtype=np.float64),
        track_id=1,
        xyz=np.asarray([0.0, 0.0, 4.0], dtype=np.float64),
        similarity=0.9,
        ratio=0.0,
        landmark_variance=0.0,
        patch_offset_confidence=0.8,
        patch_offset_sigma=0.5,
        patch_offset_applied=True,
        patch_offset_consistency_before_px=4.0,
        patch_offset_consistency_after_px=1.0,
    )
    regressed = QueryTo3DMatch(
        token_index=1,
        xy=np.asarray([20.0, 20.0], dtype=np.float64),
        track_id=2,
        xyz=np.asarray([0.2, 0.2, 5.0], dtype=np.float64),
        similarity=0.9,
        ratio=0.0,
        landmark_variance=0.0,
        patch_offset_confidence=0.8,
        patch_offset_sigma=0.5,
        patch_offset_applied=True,
        patch_offset_consistency_before_px=1.0,
        patch_offset_consistency_after_px=5.0,
    )

    weights = _fixed_pose_refinement_weights([improved, regressed], "offset_consistency_composite")

    assert weights is not None
    assert weights[0] > weights[1]
    assert weights[1] >= 0.05


def test_eval_patch_to_3d_cli_requires_candidate_bank_for_reference_visibility(tmp_path: Path) -> None:
    dim = 2
    token_path = tmp_path / "q.npz"
    np.savez_compressed(token_path, radio_final=np.zeros((dim, 1, 1), dtype=np.float32))
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
    save_selected_track_bank_npz(SelectedTrackFeatureBank(tracks={}, feature_dim=dim), bank_path)
    tracks_path = tmp_path / "tracks.jsonl"
    tracks_path.write_text("")
    pose_path = tmp_path / "poses.txt"
    pose_path.write_text(
        "Visual Landmark Dataset V1\n"
        "ImageFile, Camera Position [X Y Z W P Q R]\n\n"
        "q.png 0 0 0 1 0 0 0\n"
    )

    with pytest.raises(ValueError, match="candidate_bank is required"):
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
                "reference_visibility",
                "--default_camera",
                "1,100,100,80,80,50,50",
                "--output_jsonl",
                str(tmp_path / "rows.jsonl"),
                "--summary_json",
                str(tmp_path / "summary.json"),
            ]
        )


def test_eval_patch_to_3d_cli_supports_gt_visible_submap(tmp_path: Path) -> None:
    dim = 4
    query_map = np.zeros((dim, 1, 1), dtype=np.float32)
    query_map[:, 0, 0] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
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
                1: TrackFeature(1, np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), np.zeros((dim,), dtype=np.float32), 3, 1.0, ("ref.png",)),
                2: TrackFeature(2, np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32), np.zeros((dim,), dtype=np.float32), 3, 1.0, ("ref.png",)),
            },
            feature_dim=dim,
        ),
        bank_path,
    )
    tracks_path = tmp_path / "tracks.jsonl"
    tracks_path.write_text(
        json.dumps({"track_id": 1, "image_id": "ref.png", "point2d_idx": 0, "xy": [50.0, 50.0], "xyz": [0.0, 0.0, 5.0], "track_length": 3, "reprojection_error": 0.1}) + "\n"
        + json.dumps({"track_id": 2, "image_id": "ref.png", "point2d_idx": 1, "xy": [50.0, 50.0], "xyz": [0.0, 0.0, -5.0], "track_length": 3, "reprojection_error": 0.1}) + "\n"
    )
    pose_path = tmp_path / "poses.txt"
    pose_path.write_text(
        "Visual Landmark Dataset V1\n"
        "ImageFile, Camera Position [X Y Z W P Q R]\n\n"
        "q.png 0 0 0 1 0 0 0\n"
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
            "--query_pose_file",
            str(pose_path),
            "--submap_mode",
            "gt_visible",
            "--default_camera",
            "1,100,100,80,80,50,50",
            "--match_mode",
            "nn",
            "--top_k",
            "1",
            "--min_similarity",
            "0.5",
            "--output_jsonl",
            str(rows_path),
            "--summary_json",
            str(summary_path),
        ]
    )

    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    summary = json.loads(summary_path.read_text())
    assert rows[0]["submap_landmark_count"] == 1
    assert rows[0]["gt_visible_bank_tracks"] == 1
    assert rows[0]["submap_gt_visible_tracks"] == 1
    assert rows[0]["visible_landmark_recall"] == 1.0
    assert summary["submap"]["mode"] == "gt_visible"
    assert summary["visible_landmark_recall"]["mean"] == 1.0
