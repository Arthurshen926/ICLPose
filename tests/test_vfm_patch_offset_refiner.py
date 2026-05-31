from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.map_lifting import SelectedTrackFeatureBank, TrackFeature, save_selected_track_bank_npz
from feature_extract.vfm.patch_offset_refiner import (
    HeatmapPatchOffsetRefiner,
    HeatmapPatchOffsetRefinerConfig,
    PatchOffsetRefinerConfig,
    build_patch_offset_samples_from_rows,
    apply_predicted_patch_offsets,
    refine_matches_with_oracle_offsets,
    train_heatmap_patch_offset_refiner,
    train_patch_offset_refiner,
)
from feature_extract.vfm.query_to_3d_matching import QueryTo3DMatch
from feature_extract.vfm.tokens import TokenBankManifest, TokenBankRecord, TokenLayerSpec


def _camera() -> ColmapCamera:
    return ColmapCamera(camera_id=1, model_id=1, width=64, height=64, params=(40.0, 40.0, 32.0, 32.0))


def _pose() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def _xyz_for_xy(x: float, y: float, z: float = 4.0) -> np.ndarray:
    return np.asarray([(x - 32.0) / 40.0 * z, (y - 32.0) / 40.0 * z, z], dtype=np.float64)


def _write_offset_fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, np.ndarray]]:
    feature = np.zeros((4, 2, 2), dtype=np.float32)
    feature[:, 0, 0] = [1.0, 0.0, 0.0, 0.0]
    feature[:, 0, 1] = [0.0, 1.0, 0.0, 0.0]
    feature[:, 1, 0] = [0.0, 0.0, 1.0, 0.0]
    feature[:, 1, 1] = [0.0, 0.0, 0.0, 1.0]
    token_path = tmp_path / "q.npz"
    np.savez_compressed(token_path, radio_final=feature)
    manifest = TokenBankManifest(
        records=(
            TokenBankRecord(
                image_id="q.png",
                token_path=token_path,
                layers=(TokenLayerSpec("radio_final", "synthetic", "final", 4, 32),),
                split="train",
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
                1: TrackFeature(1, np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), np.zeros((4,), dtype=np.float32), 5, 1.0, ("r",)),
                2: TrackFeature(2, np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32), np.zeros((4,), dtype=np.float32), 5, 1.0, ("r",)),
            },
            feature_dim=4,
        ),
        bank_path,
    )
    rows = [
        {
            "query_id": "q.png",
            "token_index": 0,
            "track_id": 1,
            "xy": [0.0, 0.0],
            "xyz": _xyz_for_xy(6.0, 8.0).tolist(),
            "similarity": 0.9,
            "similarity_margin": 0.2,
            "observation_count": 5,
            "landmark_variance": 0.01,
            "landmark_reprojection_error": 0.5,
            "patch_positive_label": True,
            "pnp_inlier": True,
        },
        {
            "query_id": "q.png",
            "token_index": 1,
            "track_id": 2,
            "xy": [63.0, 0.0],
            "xyz": _xyz_for_xy(56.0, 9.0).tolist(),
            "similarity": 0.8,
            "similarity_margin": 0.1,
            "observation_count": 5,
            "landmark_variance": 0.01,
            "landmark_reprojection_error": 0.5,
            "hard_negative_label": False,
            "pnp_inlier": True,
        },
        {
            "query_id": "q.png",
            "token_index": 1,
            "track_id": 1,
            "xy": [63.0, 0.0],
            "xyz": _xyz_for_xy(8.0, 8.0).tolist(),
            "similarity": 0.4,
            "similarity_margin": 0.01,
            "observation_count": 5,
            "landmark_variance": 0.01,
            "landmark_reprojection_error": 0.5,
            "patch_positive_label": True,
            "pnp_inlier": False,
        },
    ]
    match_path = tmp_path / "matches.jsonl"
    match_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return manifest_path, bank_path, match_path, {"q.png": _pose()}


def test_build_patch_offset_samples_computes_signed_offsets_from_gt_projection(tmp_path: Path) -> None:
    manifest_path, bank_path, match_path, pose_by_query = _write_offset_fixture(tmp_path)

    samples, meta = build_patch_offset_samples_from_rows(
        match_jsonl=match_path,
        query_manifest=manifest_path,
        landmark_bank=bank_path,
        pose_by_query=pose_by_query,
        camera=_camera(),
        window_size=3,
        positive_stride=1.0,
        negative_stride=2.0,
    )

    assert samples.sample_count == 3
    assert samples.query_windows.shape == (3, 3, 3, 4)
    np.testing.assert_allclose(samples.target_offsets[0], [6.0 / 63.0, 8.0 / 63.0], atol=1e-6)
    assert samples.labels.tolist() == [1.0, 1.0, 1.0]
    assert meta["positive_count"] == 3


def test_build_patch_offset_samples_can_focus_first_pass_inliers_and_no_refine_labels(tmp_path: Path) -> None:
    manifest_path, bank_path, match_path, pose_by_query = _write_offset_fixture(tmp_path)

    samples, meta = build_patch_offset_samples_from_rows(
        match_jsonl=match_path,
        query_manifest=manifest_path,
        landmark_bank=bank_path,
        pose_by_query=pose_by_query,
        camera=_camera(),
        require_pnp_inlier=True,
        positive_label_mode="patch",
        bounded_positive_stride=0.5,
        negative_stride=99.0,
    )

    assert samples.sample_count == 2
    assert samples.labels.tolist() == [1.0, 0.0]
    assert samples.refine_labels.tolist() == [1.0, 0.0]
    assert samples.sample_weights.tolist() == [1.0, 1.0]
    assert meta["pnp_inlier_count"] == 2
    assert meta["non_refine_inlier_count"] == 1


def test_oracle_offset_refines_only_requested_inliers() -> None:
    matches = [
        QueryTo3DMatch(0, np.asarray([0.0, 0.0]), 1, _xyz_for_xy(6.0, 8.0), 0.9, 0.0, 0.0),
        QueryTo3DMatch(1, np.asarray([63.0, 0.0]), 2, _xyz_for_xy(56.0, 9.0), 0.8, 0.0, 0.0),
    ]

    refined, summary = refine_matches_with_oracle_offsets(
        matches,
        pose_w2c=_pose(),
        camera=_camera(),
        stride_px=63.0,
        inlier_mask=np.asarray([True, False]),
    )

    np.testing.assert_allclose(refined[0].xy, [6.0, 8.0], atol=1e-6)
    np.testing.assert_allclose(refined[1].xy, [63.0, 0.0], atol=1e-6)
    assert summary["refined_count"] == 1


def test_oracle_offset_can_be_free_or_bounded_by_patch_stride() -> None:
    matches = [
        QueryTo3DMatch(0, np.asarray([0.0, 0.0]), 1, _xyz_for_xy(30.0, 0.0), 0.9, 0.0, 0.0),
    ]

    bounded, bounded_summary = refine_matches_with_oracle_offsets(
        matches,
        pose_w2c=_pose(),
        camera=_camera(),
        stride_px=16.0,
        max_offset_stride=0.5,
        bound_metric="linf",
    )
    free, free_summary = refine_matches_with_oracle_offsets(
        matches,
        pose_w2c=_pose(),
        camera=_camera(),
        stride_px=16.0,
        max_offset_stride=None,
    )

    np.testing.assert_allclose(bounded[0].xy, [0.0, 0.0], atol=1e-6)
    assert bounded_summary["refined_count"] == 0
    assert bounded_summary["rejected_by_bound_count"] == 1
    np.testing.assert_allclose(free[0].xy, [30.0, 0.0], atol=1e-6)
    assert free_summary["refined_count"] == 1
    assert free_summary["bounded"] is False


def test_oracle_offset_can_require_patch_positive_matches() -> None:
    matches = [
        QueryTo3DMatch(0, np.asarray([0.0, 0.0]), 1, _xyz_for_xy(6.0, 8.0), 0.9, 0.0, 0.0),
        QueryTo3DMatch(0, np.asarray([0.0, 0.0]), 2, _xyz_for_xy(10.0, 12.0), 0.8, 0.0, 0.0),
    ]

    refined, summary = refine_matches_with_oracle_offsets(
        matches,
        pose_w2c=_pose(),
        camera=_camera(),
        stride_px=16.0,
        max_offset_stride=1.0,
        patch_positive_by_token={0: {1}},
        require_patch_positive=True,
    )

    np.testing.assert_allclose(refined[0].xy, [6.0, 8.0], atol=1e-6)
    np.testing.assert_allclose(refined[1].xy, [0.0, 0.0], atol=1e-6)
    assert summary["refined_count"] == 1
    assert summary["rejected_by_patch_positive_count"] == 1


def test_patch_offset_refiner_reduces_positive_offset_mae(tmp_path: Path) -> None:
    manifest_path, bank_path, match_path, pose_by_query = _write_offset_fixture(tmp_path)
    samples, _meta = build_patch_offset_samples_from_rows(
        match_jsonl=match_path,
        query_manifest=manifest_path,
        landmark_bank=bank_path,
        pose_by_query=pose_by_query,
        camera=_camera(),
    )

    run = train_patch_offset_refiner(
        samples,
        PatchOffsetRefinerConfig(
            feature_dim=4,
            stats_dim=samples.match_stats.shape[1],
            hidden_dim=16,
            steps=120,
            batch_size=2,
            lr=0.05,
            confidence_loss_weight=0.1,
            device="cpu",
            seed=0,
        ),
    )

    assert run.summary.final_positive_offset_mae_px < run.summary.initial_positive_offset_mae_px
    assert run.summary.final_loss < run.summary.initial_loss


def test_heatmap_patch_offset_refiner_outputs_bounded_offset_and_uncertainty() -> None:
    model = HeatmapPatchOffsetRefiner(feature_dim=4, stats_dim=6, window_size=3, hidden_dim=16, max_offset_stride=0.5, bin_count=4)

    out = model(
        query_windows=np.zeros((2, 3, 3, 4), dtype=np.float32),
        landmark_features=np.zeros((2, 4), dtype=np.float32),
        match_stats=np.zeros((2, 6), dtype=np.float32),
    )

    assert out["offset"].shape == (2, 2)
    assert out["heatmap_logits"].shape == (2, 16)
    assert out["log_sigma"].shape == (2, 1)
    assert np.all(np.abs(out["offset"].detach().numpy()) <= 0.5 + 1e-6)


def test_heatmap_patch_offset_refiner_reduces_positive_offset_mae(tmp_path: Path) -> None:
    manifest_path, bank_path, match_path, pose_by_query = _write_offset_fixture(tmp_path)
    samples, _meta = build_patch_offset_samples_from_rows(
        match_jsonl=match_path,
        query_manifest=manifest_path,
        landmark_bank=bank_path,
        pose_by_query=pose_by_query,
        camera=_camera(),
        require_pnp_inlier=True,
        positive_label_mode="patch",
        bounded_positive_stride=0.5,
        negative_stride=99.0,
    )

    run = train_heatmap_patch_offset_refiner(
        samples,
        HeatmapPatchOffsetRefinerConfig(
            feature_dim=4,
            stats_dim=samples.stats_dim,
            hidden_dim=32,
            steps=100,
            batch_size=2,
            lr=0.005,
            confidence_loss_weight=0.1,
            residual_loss_weight=1.0,
            eval_split_fraction=0.0,
            device="cpu",
            seed=0,
            bin_count=4,
        ),
    )

    assert run.summary.final_positive_offset_mae_px < run.summary.initial_positive_offset_mae_px


def test_apply_predicted_patch_offsets_can_keep_high_sigma_matches_at_patch_center() -> None:
    matches = [
        QueryTo3DMatch(0, np.asarray([0.0, 0.0]), 1, _xyz_for_xy(6.0, 8.0), 0.9, 0.0, 0.0),
        QueryTo3DMatch(1, np.asarray([63.0, 0.0]), 2, _xyz_for_xy(56.0, 9.0), 0.8, 0.0, 0.0),
    ]

    refined, summary = apply_predicted_patch_offsets(
        matches,
        offsets=np.asarray([[0.25, 0.0], [0.25, 0.0]], dtype=np.float32),
        confidences=np.asarray([0.9, 0.9], dtype=np.float32),
        stride_px=16.0,
        sigmas=np.asarray([0.2, 2.0], dtype=np.float32),
        max_sigma=1.0,
        confidence_threshold=0.5,
    )

    np.testing.assert_allclose(refined[0].xy, [4.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(refined[1].xy, [63.0, 0.0], atol=1e-6)
    assert summary["refined_count"] == 1
    assert summary["rejected_by_sigma_count"] == 1
