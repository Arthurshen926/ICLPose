import numpy as np
import torch


def _toy_samples():
    from feature_extract.vfm.patch_selector_training import PatchSelectorTrainingSet

    queries = np.asarray(
        [
            [1.0, 0.0, 0.2, 0.0],
            [0.0, 1.0, 0.0, 0.2],
            [1.0, 0.0, -0.2, 0.0],
            [0.0, 1.0, 0.0, -0.2],
        ],
        dtype=np.float32,
    )
    positives = np.asarray(
        [
            [[1.0, 0.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    negatives = np.asarray(
        [
            [[0.0, 1.0, 0.2, 0.0], [0.0, 1.0, -0.2, 0.0]],
            [[1.0, 0.0, 0.0, 0.2], [1.0, 0.0, 0.0, -0.2]],
            [[0.0, 1.0, -0.2, 0.0], [0.0, 1.0, 0.2, 0.0]],
            [[1.0, 0.0, 0.0, -0.2], [1.0, 0.0, 0.0, 0.2]],
        ],
        dtype=np.float32,
    )
    return PatchSelectorTrainingSet(
        query_features=queries,
        positive_features=positives,
        positive_mask=np.ones((4, 1), dtype=bool),
        negative_features=negatives,
    )


def test_residual_gated_selector_outputs_normalized_descriptors_and_pair_logits():
    from feature_extract.vfm.patch_selector_training import ResidualGatedPatchSelector

    selector = ResidualGatedPatchSelector(
        input_dim=8,
        output_dim=4,
        residual_hidden_dim=6,
        group_size=2,
    )
    features = torch.randn(5, 8)
    descriptors = selector(features)
    logits = selector.pairwise_inlier_logit(descriptors[:2], descriptors[2:4])

    assert descriptors.shape == (5, 4)
    assert logits.shape == (2,)
    assert selector.group_gates().shape == (4,)
    torch.testing.assert_close(torch.linalg.norm(descriptors, dim=1), torch.ones(5), atol=1e-5, rtol=1e-5)


def test_safe_selector_training_reduces_loss_and_reports_active_groups():
    from feature_extract.vfm.patch_selector_training import (
        SafePatchSelectorTrainingConfig,
        train_safe_patch_selector,
    )

    run = train_safe_patch_selector(
        _toy_samples(),
        SafePatchSelectorTrainingConfig(
            output_dim=2,
            residual_hidden_dim=4,
            steps=80,
            batch_size=4,
            lr=0.05,
            seed=4,
            device="cpu",
            eval_split_fraction=0.0,
            group_size=2,
            hard_gate_keep_fraction=0.5,
            inlier_loss_weight=0.2,
        ),
    )

    assert run.summary.final_loss < run.summary.initial_loss
    assert run.summary.train_top1_acc >= 0.75
    assert run.summary.inlier_train_accuracy >= 0.75
    assert run.summary.group_count == 2
    assert run.summary.active_group_count == 1
    encoded = run.encode_rows(_toy_samples().query_features, device="cpu", batch_size=2)
    assert encoded.shape == (4, 2)
    np.testing.assert_allclose(np.linalg.norm(encoded, axis=1), np.ones(4), atol=1e-5)


def test_safe_selector_checkpoint_round_trip(tmp_path):
    from feature_extract.vfm.patch_selector_training import (
        SafePatchSelectorTrainingConfig,
        load_safe_patch_selector_checkpoint,
        save_safe_patch_selector_checkpoint,
        train_safe_patch_selector,
    )

    run = train_safe_patch_selector(
        _toy_samples(),
        SafePatchSelectorTrainingConfig(
            output_dim=2,
            residual_hidden_dim=4,
            steps=5,
            batch_size=4,
            seed=5,
            device="cpu",
            eval_split_fraction=0.0,
            group_size=2,
        ),
    )
    path = tmp_path / "selector.pt"
    save_safe_patch_selector_checkpoint(run, path)
    loaded = load_safe_patch_selector_checkpoint(path, device="cpu")

    before = run.encode_rows(_toy_samples().query_features, device="cpu", batch_size=4)
    after = loaded.encode_rows(_toy_samples().query_features, device="cpu", batch_size=4)
    np.testing.assert_allclose(after, before, atol=1e-6)


def test_stage_c2_cli_trains_checkpoint_and_exports_descriptor_bank(tmp_path):
    from feature_extract.tools.vfm.train_stage_c2_safe_selector import main
    from feature_extract.vfm.map_lifting import (
        SelectedTrackFeatureBank,
        TrackFeature,
        load_selected_track_bank_npz,
        save_selected_track_bank_npz,
    )
    from feature_extract.vfm.patch_selector_training import save_patch_selector_training_set_npz
    from feature_extract.vfm.tokens import TokenBankManifest, TokenBankRecord, TokenLayerSpec

    sample_cache = tmp_path / "samples.npz"
    save_patch_selector_training_set_npz(_toy_samples(), sample_cache)

    token_path = tmp_path / "tokens" / "query.npz"
    token_path.parent.mkdir()
    feature_map = np.zeros((4, 1, 2), dtype=np.float32)
    feature_map[:, 0, 0] = np.asarray([1.0, 0.0, 0.2, 0.0], dtype=np.float32)
    feature_map[:, 0, 1] = np.asarray([0.0, 1.0, 0.0, 0.2], dtype=np.float32)
    np.savez_compressed(token_path, radio_final=feature_map)
    manifest = TokenBankManifest(
        records=(
            TokenBankRecord(
                image_id="query.png",
                token_path=token_path,
                layers=(TokenLayerSpec(name="radio_final", model="toy", layer="final", channels=4, stride=2),),
                split="test",
                scene="toy",
            ),
        )
    )
    manifest_path = tmp_path / "manifest.json"
    manifest.to_json(manifest_path)

    bank_path = tmp_path / "bank.npz"
    save_selected_track_bank_npz(
        SelectedTrackFeatureBank(
            tracks={
                1: TrackFeature(1, np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), np.zeros(4, dtype=np.float32), 3, 1.0, ("a",)),
                2: TrackFeature(2, np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32), np.zeros(4, dtype=np.float32), 3, 1.0, ("b",)),
            },
            feature_dim=4,
        ),
        bank_path,
    )

    output_dir = tmp_path / "out"
    main(
        [
            "--sample_cache",
            str(sample_cache),
            "--query_manifest",
            str(manifest_path),
            "--landmark_bank",
            str(bank_path),
            "--output_dim",
            "2",
            "--residual_hidden_dim",
            "4",
            "--steps",
            "3",
            "--batch_size",
            "2",
            "--group_size",
            "2",
            "--hard_gate_keep_fraction",
            "0.5",
            "--output_model",
            str(output_dir / "selector.pt"),
            "--summary_json",
            str(output_dir / "summary.json"),
            "--output_query_dir",
            str(output_dir / "tokens"),
            "--output_query_manifest",
            str(output_dir / "manifest.json"),
            "--output_landmark_bank",
            str(output_dir / "bank.npz"),
        ]
    )

    assert (output_dir / "selector.pt").exists()
    compressed_manifest = TokenBankManifest.from_json(output_dir / "manifest.json")
    assert compressed_manifest.records[0].layers[0].channels == 2
    compressed_bank = load_selected_track_bank_npz(output_dir / "bank.npz")
    assert compressed_bank.feature_dim == 2


def test_patch_matcher_can_rescore_cosine_topk_with_pairwise_inlier_logprob():
    from feature_extract.vfm.patch_to_3d_matching import PatchTo3DMatchingConfig, match_query_patches_to_landmarks
    from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex

    class DummyPairwiseScorer:
        def score_pairs(self, query_descriptors, landmark_descriptors):
            del query_descriptors
            return np.where(np.asarray(landmark_descriptors)[:, 1] > 0.1, 5.0, -5.0).astype(np.float32)

    query_map = np.zeros((2, 1, 1), dtype=np.float32)
    query_map[:, 0, 0] = np.asarray([1.0, 0.0], dtype=np.float32)
    index = LandmarkMapIndex(
        track_ids=np.asarray([1, 2], dtype=np.int64),
        xyz=np.asarray([[0.0, 0.0, 5.0], [0.0, 0.0, 6.0]], dtype=np.float64),
        features=np.asarray([[1.0, 0.0], [0.8, 0.6]], dtype=np.float32),
        mean_variances=np.zeros((2,), dtype=np.float32),
        observation_counts=np.ones((2,), dtype=np.int64) * 3,
        observation_image_ids=(("a",), ("b",)),
    )

    matches = match_query_patches_to_landmarks(
        query_map,
        index,
        PatchTo3DMatchingConfig(
            top_k=2,
            match_mode="nn",
            ratio_threshold=None,
            min_similarity=0.0,
            match_score_mode="similarity_pairwise",
        ),
        image_width=100,
        image_height=100,
        pairwise_inlier_scorer=DummyPairwiseScorer(),
        pairwise_inlier_weight=1.0,
    )

    assert len(matches) == 1
    assert matches[0].track_id == 2
    assert matches[0].pairwise_inlier_logit > 0.0
    assert matches[0].pairwise_inlier_logprob > -0.1


def test_safe_pairwise_scorer_loads_checkpoint_and_scores_descriptor_pairs(tmp_path):
    from feature_extract.vfm.patch_selector_training import (
        SafePairwiseInlierScorer,
        SafePatchSelectorTrainingConfig,
        save_safe_patch_selector_checkpoint,
        train_safe_patch_selector,
    )

    run = train_safe_patch_selector(
        _toy_samples(),
        SafePatchSelectorTrainingConfig(
            output_dim=2,
            residual_hidden_dim=4,
            steps=5,
            batch_size=4,
            seed=6,
            device="cpu",
            eval_split_fraction=0.0,
            group_size=2,
        ),
    )
    checkpoint = tmp_path / "selector.pt"
    save_safe_patch_selector_checkpoint(run, checkpoint)
    scorer = SafePairwiseInlierScorer.from_checkpoint(checkpoint, device="cpu", batch_size=2)
    descriptors = run.encode_rows(_toy_samples().query_features, device="cpu", batch_size=4)

    logits = scorer.score_pairs(descriptors[:2], descriptors[2:4])

    assert logits.shape == (2,)
    assert np.all(np.isfinite(logits))


def test_stage_c2_export_cli_overrides_active_group_fraction(tmp_path):
    from feature_extract.tools.vfm.export_stage_c2_safe_selector import main
    from feature_extract.vfm.map_lifting import (
        SelectedTrackFeatureBank,
        TrackFeature,
        load_selected_track_bank_npz,
        save_selected_track_bank_npz,
    )
    from feature_extract.vfm.patch_selector_training import (
        SafePatchSelectorTrainingConfig,
        save_safe_patch_selector_checkpoint,
        train_safe_patch_selector,
    )
    from feature_extract.vfm.tokens import TokenBankManifest, TokenBankRecord, TokenLayerSpec

    run = train_safe_patch_selector(
        _toy_samples(),
        SafePatchSelectorTrainingConfig(
            output_dim=2,
            residual_hidden_dim=4,
            steps=5,
            batch_size=4,
            seed=8,
            device="cpu",
            eval_split_fraction=0.0,
            group_size=2,
            hard_gate_keep_fraction=1.0,
        ),
    )
    checkpoint = tmp_path / "selector.pt"
    save_safe_patch_selector_checkpoint(run, checkpoint)

    token_path = tmp_path / "tokens" / "query.npz"
    token_path.parent.mkdir()
    np.savez_compressed(token_path, radio_final=np.ones((4, 1, 1), dtype=np.float32))
    manifest_path = tmp_path / "manifest.json"
    TokenBankManifest(
        records=(
            TokenBankRecord(
                image_id="query.png",
                token_path=token_path,
                layers=(TokenLayerSpec(name="radio_final", model="toy", layer="final", channels=4, stride=2),),
                split="test",
                scene="toy",
            ),
        )
    ).to_json(manifest_path)
    bank_path = tmp_path / "bank.npz"
    save_selected_track_bank_npz(
        SelectedTrackFeatureBank(
            tracks={
                1: TrackFeature(1, np.ones(4, dtype=np.float32), np.zeros(4, dtype=np.float32), 3, 1.0, ("a",)),
            },
            feature_dim=4,
        ),
        bank_path,
    )
    output_dir = tmp_path / "out"

    main(
        [
            "--checkpoint",
            str(checkpoint),
            "--query_manifest",
            str(manifest_path),
            "--landmark_bank",
            str(bank_path),
            "--hard_gate_keep_fraction",
            "0.5",
            "--output_query_dir",
            str(output_dir / "tokens"),
            "--output_query_manifest",
            str(output_dir / "manifest.json"),
            "--output_landmark_bank",
            str(output_dir / "bank.npz"),
            "--summary_json",
            str(output_dir / "summary.json"),
        ]
    )

    compressed_bank = load_selected_track_bank_npz(output_dir / "bank.npz")
    assert compressed_bank.feature_dim == 2
    summary = __import__("json").loads((output_dir / "summary.json").read_text())
    assert summary["active_group_count"] == 1
    assert summary["group_count"] == 2
