from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.map_lifting import SelectedTrackFeatureBank, TrackFeature, save_selected_track_bank_npz
from feature_extract.vfm.mnn_consistent_descriptor_selection import (
    MNNConsistentDescriptorSelectionConfig,
    MNNConsistentDescriptorTrainingSet,
    build_mnn_consistent_descriptor_samples,
    train_mnn_consistent_descriptor_selector,
)
from feature_extract.vfm.tokens import TokenBankManifest, TokenBankRecord, TokenLayerSpec


def _write_mnn_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    query_map = np.zeros((3, 1, 2), dtype=np.float32)
    query_map[:, 0, 0] = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    query_map[:, 0, 1] = np.asarray([0.2, 1.0, 0.0], dtype=np.float32)
    token_path = tmp_path / "query_tokens.npz"
    np.savez_compressed(token_path, radio_final=query_map)
    manifest = TokenBankManifest(
        records=(
            TokenBankRecord(
                image_id="q.png",
                token_path=token_path,
                layers=(TokenLayerSpec("radio_final", "synthetic", "final", 3, 16),),
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
                1: TrackFeature(1, np.asarray([1.0, 0.0, 0.0], dtype=np.float32), np.zeros((3,), dtype=np.float32), 8, 1.0, ("r",)),
                2: TrackFeature(2, np.asarray([0.0, 1.0, 0.0], dtype=np.float32), np.zeros((3,), dtype=np.float32), 8, 1.0, ("r",)),
                3: TrackFeature(3, np.asarray([0.7, 0.7, 0.0], dtype=np.float32), np.zeros((3,), dtype=np.float32), 2, 1.0, ("r",)),
            },
            feature_dim=3,
        ),
        bank_path,
    )

    rows = [
        {"query_id": "q.png", "token_index": 0, "track_id": 1, "strong_positive_label": True, "gt_reproj_error_stride": 0.0, "teacher_score": 0.95, "similarity": 0.9},
        {"query_id": "q.png", "token_index": 0, "track_id": 3, "hard_negative_label": True, "baseline_ransac_inlier": True, "gt_reproj_error_stride": 4.0, "teacher_score": 0.05, "similarity": 0.85},
        {"query_id": "q.png", "token_index": 1, "track_id": 2, "patch_positive_label": True, "gt_reproj_error_stride": 0.5, "teacher_score": 0.9, "similarity": 0.88},
        {"query_id": "q.png", "token_index": 1, "track_id": 3, "hard_negative_label": True, "baseline_ransac_inlier": True, "gt_reproj_error_stride": 3.5, "teacher_score": 0.04, "similarity": 0.82},
    ]
    match_path = tmp_path / "matches.jsonl"
    match_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return manifest_path, bank_path, match_path


def test_build_mnn_samples_create_soft_labels_and_wrong_pose_weights(tmp_path: Path) -> None:
    manifest_path, bank_path, match_path = _write_mnn_fixture(tmp_path)

    samples, meta = build_mnn_consistent_descriptor_samples(
        match_jsonl=match_path,
        query_manifest=manifest_path,
        landmark_bank=bank_path,
        max_tokens_per_query=4,
        max_landmarks_per_query=4,
        soft_label_sigma_stride=1.0,
    )

    assert samples.group_count == 1
    assert samples.query_features.shape == (1, 2, 3)
    assert samples.landmark_features.shape == (1, 3, 3)
    assert samples.valid_mask.sum() == 4
    assert samples.soft_labels[0].max() == 1.0
    assert 0.0 < samples.soft_labels[0, 1, 1] < 1.0
    assert samples.wrong_pose_negative_weights.sum() == 2.0
    assert meta["wrong_pose_negative_count"] == 2


def test_mnn_consistent_selector_improves_dual_top1_on_ambiguous_matches() -> None:
    query = np.asarray([[[1.0, 0.1], [1.0, -0.1]]], dtype=np.float32)
    landmarks = np.asarray([[[1.0, 0.0], [1.0, 0.2]]], dtype=np.float32)
    valid = np.ones((1, 2, 2), dtype=bool)
    hard_labels = np.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
    soft_labels = hard_labels.copy()
    teacher = hard_labels * 0.9 + 0.05

    samples = MNNConsistentDescriptorTrainingSet(
        query_features=query,
        landmark_features=landmarks,
        valid_mask=valid,
        hard_labels=hard_labels,
        soft_labels=soft_labels,
        pair_weights=valid.astype(np.float32),
        teacher_scores=teacher,
        wrong_pose_negative_weights=np.zeros((1, 2, 2), dtype=np.float32),
    )

    run = train_mnn_consistent_descriptor_selector(
        samples,
        MNNConsistentDescriptorSelectionConfig(
            output_dim=2,
            hidden_dim=16,
            steps=250,
            batch_size=1,
            lr=0.01,
            dual_loss_weight=1.0,
            soft_reproj_loss_weight=1.0,
            teacher_distill_loss_weight=1.0,
            anchor_loss_weight=0.0,
            seed=0,
        ),
    )

    assert run.summary.raw_train_dual_top1_acc < 1.0
    assert run.summary.train_dual_top1_acc > run.summary.raw_train_dual_top1_acc
    assert run.summary.final_loss < run.summary.initial_loss


def test_stage_c210_cli_writes_summary(tmp_path: Path) -> None:
    from feature_extract.tools.vfm.train_stage_c210_mnn_consistent_descriptor_selection import main

    manifest_path, bank_path, match_path = _write_mnn_fixture(tmp_path)
    summary_path = tmp_path / "summary.json"
    model_path = tmp_path / "model.pt"

    main(
        [
            "--match_jsonl",
            str(match_path),
            "--query_manifest",
            str(manifest_path),
            "--landmark_bank",
            str(bank_path),
            "--output_model",
            str(model_path),
            "--summary_json",
            str(summary_path),
            "--output_dim",
            "3",
            "--steps",
            "12",
            "--batch_size",
            "1",
            "--lr",
            "0.05",
        ]
    )

    summary = json.loads(summary_path.read_text())
    assert summary["stage"] == "stage_c210_mnn_consistent_pose_aware_descriptor_selection"
    assert summary["sample_summary"]["group_count"] == 1
    assert np.isfinite(summary["training"]["train_dual_top1_acc"])
    assert model_path.exists()
