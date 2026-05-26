import json

import numpy as np
import pytest
import torch

from feature_extract.tools.vfm.train_dense_selector import main as train_dense_cli_main
from feature_extract.tools.vfm.build_dense_sample_cache import main as build_dense_sample_cache_cli_main
from feature_extract.vfm import dense_selector_training as dense_training
from feature_extract.vfm.dense_selector_training import (
    DenseSelectorTrainingConfig,
    _DenseQueryGroup,
    _basin_labels_for_group,
    _build_dense_groups,
    _hard_negative_margin_loss,
    _load_sampled_dense_feature_cache,
    _load_sampled_dense_feature,
    _loss_for_groups,
    _normalize_dense_groups,
    _pose_cost_normalizer,
    _score_group,
    _score_groups_batched,
    _selected_descriptor,
    _stable_spatial_seed,
    _write_sampled_dense_feature_cache,
    run_dense_selector_training,
    train_dense_selector,
)
from feature_extract.vfm.hypotheses import CandidateHypothesis, PoseCost
from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.map_lifting import TrackObservation
from feature_extract.vfm.protocols import ProtocolKind
from feature_extract.vfm.selector import LocalizableFeatureSelector
from feature_extract.vfm.tokens import (
    TokenBankManifest,
    TokenBankRecord,
    TokenLayerSpec,
    compute_file_sha256,
    write_npz_token_record,
)


def _write_token(tmp_path, image_id, feature, split):
    path = tmp_path / f"{image_id.replace('/', '__')}.npz"
    write_npz_token_record(path, {"radio_final": np.asarray(feature, dtype=np.float16)})
    return TokenBankRecord(
        image_id=image_id,
        token_path=path,
        layers=(TokenLayerSpec("radio_final", "synthetic", "final", feature.shape[0], 1),),
        split=split,
        scene="Synthetic",
        checksum=compute_file_sha256(path),
    )


def _synthetic_dense_problem(tmp_path, query_count=12, candidates_per_query=3, channels=8):
    rng = np.random.default_rng(13)
    query_records = []
    map_records = []
    candidates = []
    height, width = 3, 3

    for query_idx in range(query_count):
        query_id = f"q{query_idx:02d}.png"
        signal = rng.normal(size=(2, 1, 1)).astype(np.float32)
        query = rng.normal(scale=0.05, size=(channels, height, width)).astype(np.float32)
        query[:2] = signal + rng.normal(scale=0.02, size=(2, height, width))
        query[2:] = rng.normal(loc=1.0, scale=0.04, size=(channels - 2, height, width))
        query_records.append(_write_token(tmp_path, query_id, query, "test"))

        for cand_idx in range(candidates_per_query):
            ref_id = f"m{query_idx:02d}_{cand_idx}.png"
            feature = rng.normal(scale=0.25, size=(channels, height, width)).astype(np.float32)
            if cand_idx == 0:
                feature[:2] = signal + rng.normal(scale=0.02, size=(2, height, width))
                feature[2:] = rng.normal(loc=-1.0, scale=0.05, size=(channels - 2, height, width))
                pose_error = PoseCost(0.03, 1.0)
            else:
                feature[:2] = -signal + rng.normal(scale=0.02, size=(2, height, width))
                feature[2:] = query[2:] + rng.normal(scale=0.02, size=(channels - 2, height, width))
                pose_error = PoseCost(1.0 + cand_idx, 20.0)
            map_records.append(_write_token(tmp_path, ref_id, feature, "train"))
            candidates.append(
                CandidateHypothesis(
                    query_id=query_id,
                    candidate_id=f"{query_id}:{cand_idx}",
                    candidate_type="reference_pose",
                    reference_image=ref_id,
                    pose_error=pose_error,
                )
            )

    return (
        CandidateHypothesisBank.from_candidates(
            protocol_name="synthetic_dense_selector",
            protocol_kind=ProtocolKind.REFERENCE_POSE,
            candidates=candidates,
        ),
        TokenBankManifest(records=tuple(query_records)),
        TokenBankManifest(records=tuple(map_records)),
    )


def _synthetic_track_observations(track_count=6, obs_per_track=3, channels=8):
    rng = np.random.default_rng(23)
    observations = []
    for track_id in range(track_count):
        signal = rng.normal(size=(2,)).astype(np.float32)
        for obs_idx in range(obs_per_track):
            feature = rng.normal(scale=0.1, size=(channels,)).astype(np.float32)
            feature[:2] = signal + rng.normal(scale=0.02, size=(2,))
            feature[2] = 1.0 if obs_idx == 0 else -1.0
            observations.append(
                TrackObservation(
                    track_id=track_id,
                    image_id=f"track_{track_id}_{obs_idx}.png",
                    feature=feature,
                    visible=True,
                    geometry_valid=True,
                    utility=1.0 if obs_idx == 0 else 0.2,
                )
            )
    return observations


def _write_synthetic_colmap_track_observations(path, image_ids):
    rows = []
    for track_id in range(2):
        for obs_idx, image_id in enumerate(image_ids[track_id * 2 : track_id * 2 + 2]):
            rows.append(
                {
                    "track_id": 100 + track_id,
                    "image_id": image_id,
                    "point2d_idx": obs_idx,
                    "xy": [1.0 + obs_idx, 1.0],
                    "xyz": [float(track_id), 0.0, 1.0],
                    "track_length": 2,
                    "reprojection_error": 0.1 + 0.1 * obs_idx,
                    "camera_id": 1,
                    "image_width": 3,
                    "image_height": 3,
                }
            )
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def test_dense_selector_training_learns_from_dense_token_maps(tmp_path):
    bank, query_manifest, map_manifest = _synthetic_dense_problem(tmp_path)

    summary = train_dense_selector(
        bank,
        query_manifest,
        map_manifest,
        DenseSelectorTrainingConfig(
            steps=80,
            batch_size=4,
            output_dim=4,
            group_size=2,
            lr=0.03,
            eval_split_fraction=0.25,
            seed=5,
            device="cpu",
            layer_name="radio_final",
            spatial_samples=0,
        ),
    )

    assert summary.query_count == 12
    assert summary.train_query_count == 9
    assert summary.eval_query_count == 3
    assert summary.initial_loss > summary.final_loss
    assert summary.train_top1_acc >= 0.8
    assert summary.eval_top1_acc >= 0.8
    assert summary.raw_eval_top1_acc < summary.eval_top1_acc


def test_dense_selector_training_cli_writes_json_and_checkpoint(tmp_path):
    bank, query_manifest, map_manifest = _synthetic_dense_problem(tmp_path)
    bank_path = tmp_path / "candidates.jsonl"
    query_path = tmp_path / "query_manifest.json"
    map_path = tmp_path / "map_manifest.json"
    output_path = tmp_path / "result.json"
    checkpoint_path = tmp_path / "selector.pt"
    bank.to_jsonl(bank_path)
    query_manifest.to_json(query_path)
    map_manifest.to_json(map_path)

    train_dense_cli_main(
        [
            "--bank",
            str(bank_path),
            "--query_manifest",
            str(query_path),
            "--map_manifest",
            str(map_path),
            "--layer_name",
            "radio_final",
            "--steps",
            "70",
            "--batch_size",
            "4",
            "--output_dim",
            "4",
            "--group_size",
            "2",
            "--lr",
            "0.03",
            "--eval_fraction",
            "0.25",
            "--seed",
            "5",
            "--device",
            "cpu",
            "--output",
            str(output_path),
            "--checkpoint",
            str(checkpoint_path),
        ]
    )

    payload = json.loads(output_path.read_text())
    assert payload["config"]["layer_name"] == "radio_final"
    assert payload["inputs"]["protocol_name"] == "synthetic_dense_selector"
    assert payload["inputs"]["protocol_kind"] == "reference_pose"
    assert payload["inputs"]["candidate_count"] == 36
    assert payload["inputs"]["input_files"]["candidate_bank"]["sha256"]
    assert payload["inputs"]["input_files"]["query_manifest"]["sha256"]
    assert payload["result"]["query_count"] == 12
    assert payload["result"]["eval_top1_acc"] >= 0.8
    assert checkpoint_path.exists()


def test_dense_selector_training_cli_accepts_track_supervision(tmp_path):
    bank, query_manifest, map_manifest = _synthetic_dense_problem(tmp_path, query_count=4, candidates_per_query=3)
    bank_path = tmp_path / "candidates.jsonl"
    query_path = tmp_path / "query_manifest.json"
    map_path = tmp_path / "map_manifest.json"
    track_path = tmp_path / "tracks.jsonl"
    output_path = tmp_path / "result.json"
    checkpoint_path = tmp_path / "selector.pt"
    bank.to_jsonl(bank_path)
    query_manifest.to_json(query_path)
    map_manifest.to_json(map_path)
    _write_synthetic_colmap_track_observations(track_path, [record.image_id for record in map_manifest.records[:4]])

    train_dense_cli_main(
        [
            "--bank",
            str(bank_path),
            "--query_manifest",
            str(query_path),
            "--map_manifest",
            str(map_path),
            "--track_observations",
            str(track_path),
            "--track_supervision_weight",
            "0.5",
            "--track_batch_size",
            "2",
            "--track_token_manifest",
            str(map_path),
            "--steps",
            "4",
            "--batch_size",
            "2",
            "--output_dim",
            "4",
            "--group_size",
            "2",
            "--seed",
            "5",
            "--device",
            "cpu",
            "--output",
            str(output_path),
            "--checkpoint",
            str(checkpoint_path),
        ]
    )

    payload = json.loads(output_path.read_text())
    assert payload["config"]["track_supervision_weight"] == pytest.approx(0.5)
    assert payload["inputs"]["track_supervision"]["colmap_observation_count"] == 4
    assert payload["inputs"]["track_supervision"]["sampled_observation_count"] == 4
    assert payload["inputs"]["input_files"]["track_observations"]["sha256"]
    assert payload["inputs"]["input_files"]["track_token_manifest"]["sha256"]
    assert payload["result"]["track_supervision_track_count"] == 2
    assert payload["result"]["track_supervision_observation_count"] == 4
    assert checkpoint_path.exists()


def test_dense_selector_training_cli_limits_expensive_diagnostics_without_limiting_training_pool(tmp_path):
    bank, query_manifest, map_manifest = _synthetic_dense_problem(tmp_path)
    bank_path = tmp_path / "candidates.jsonl"
    query_path = tmp_path / "query_manifest.json"
    map_path = tmp_path / "map_manifest.json"
    output_path = tmp_path / "result.json"
    bank.to_jsonl(bank_path)
    query_manifest.to_json(query_path)
    map_manifest.to_json(map_path)

    train_dense_cli_main(
        [
            "--bank",
            str(bank_path),
            "--query_manifest",
            str(query_path),
            "--map_manifest",
            str(map_path),
            "--steps",
            "2",
            "--batch_size",
            "4",
            "--output_dim",
            "4",
            "--group_size",
            "2",
            "--eval_fraction",
            "0.25",
            "--diagnostic_query_limit",
            "2",
            "--seed",
            "5",
            "--device",
            "cpu",
            "--output",
            str(output_path),
        ]
    )

    payload = json.loads(output_path.read_text())
    assert payload["result"]["query_count"] == 12
    assert payload["result"]["train_query_count"] == 9
    assert payload["result"]["eval_query_count"] == 3
    assert payload["result"]["train_diagnostic_query_count"] == 2
    assert payload["result"]["eval_diagnostic_query_count"] == 2


def test_dense_selector_training_cli_infers_warm_start_dims_when_omitted(tmp_path):
    bank, query_manifest, map_manifest = _synthetic_dense_problem(tmp_path)
    bank_path = tmp_path / "candidates.jsonl"
    query_path = tmp_path / "query_manifest.json"
    map_path = tmp_path / "map_manifest.json"
    output_path = tmp_path / "result.json"
    warm_path = tmp_path / "warm_start.pt"
    checkpoint_path = tmp_path / "selector.pt"
    bank.to_jsonl(bank_path)
    query_manifest.to_json(query_path)
    map_manifest.to_json(map_path)
    selector = LocalizableFeatureSelector(input_dim=8, output_dim=4, group_size=2)
    torch.save(selector.state_dict(), warm_path)

    train_dense_cli_main(
        [
            "--bank",
            str(bank_path),
            "--query_manifest",
            str(query_path),
            "--map_manifest",
            str(map_path),
            "--layer_name",
            "radio_final",
            "--steps",
            "1",
            "--batch_size",
            "2",
            "--lr",
            "1e-12",
            "--eval_fraction",
            "0.25",
            "--seed",
            "5",
            "--device",
            "cpu",
            "--init_checkpoint",
            str(warm_path),
            "--output",
            str(output_path),
            "--checkpoint",
            str(checkpoint_path),
        ]
    )

    payload = json.loads(output_path.read_text())
    saved_state = torch.load(checkpoint_path, map_location="cpu")
    assert payload["config"]["output_dim"] == 4
    assert payload["config"]["group_size"] == 2
    assert tuple(saved_state["projection.weight"].shape) == (4, 8, 1, 1)


def test_dense_selector_training_can_warm_start_from_selector_checkpoint(tmp_path):
    bank, query_manifest, map_manifest = _synthetic_dense_problem(tmp_path)
    checkpoint_path = tmp_path / "warm_start.pt"
    initial_selector = LocalizableFeatureSelector(input_dim=8, output_dim=4, group_size=2)
    with torch.no_grad():
        initial_selector.group_logits.fill_(4.0)
        initial_selector.projection.weight.zero_()
        initial_selector.projection.weight[0, 0, 0, 0] = 1.0
    torch.save(initial_selector.state_dict(), checkpoint_path)

    run = run_dense_selector_training(
        bank,
        query_manifest,
        map_manifest,
        DenseSelectorTrainingConfig(
            steps=1,
            batch_size=4,
            output_dim=4,
            group_size=2,
            lr=1e-12,
            eval_split_fraction=0.25,
            seed=5,
            device="cpu",
            layer_name="radio_final",
            spatial_samples=0,
            init_checkpoint=str(checkpoint_path),
        ),
    )

    assert torch.allclose(run.selector.group_logits, torch.full_like(run.selector.group_logits, 4.0), atol=1e-6)
    loaded_weight = float(run.selector.projection.weight[0, 0, 0, 0].detach().cpu())
    assert loaded_weight == pytest.approx(1.0)


def test_dense_selector_training_spatial_seed_is_stable():
    seed_a = _stable_spatial_seed("seq1/frame00001.png", 7)
    seed_b = _stable_spatial_seed("seq1/frame00001.png", 7)
    seed_c = _stable_spatial_seed("seq1/frame00002.png", 7)

    assert seed_a == seed_b
    assert seed_a != seed_c
    assert 0 <= seed_a < 2**32


def test_dense_selector_training_cache_keeps_sampled_feature_shape(tmp_path):
    token_path = tmp_path / "feature.npz"
    write_npz_token_record(
        token_path,
        {"radio_final": np.arange(8 * 4 * 4, dtype=np.float32).reshape(8, 4, 4)},
    )
    _load_sampled_dense_feature.cache_clear()

    sampled = _load_sampled_dense_feature(str(token_path), "radio_final", 3, 11)

    assert sampled.shape == (8, 3, 1)
    assert _load_sampled_dense_feature.cache_info().currsize == 1


def test_dense_selector_training_sample_cache_has_bounded_memory_budget():
    assert _load_sampled_dense_feature.cache_info().maxsize <= 512


def test_dense_selector_batched_scores_match_individual_group_scores(tmp_path):
    bank, query_manifest, map_manifest = _synthetic_dense_problem(tmp_path, query_count=4, candidates_per_query=3)
    groups = _build_dense_groups(bank, query_manifest, map_manifest)
    groups = _normalize_dense_groups(groups, _pose_cost_normalizer(groups))
    config = DenseSelectorTrainingConfig(
        steps=1,
        batch_size=4,
        output_dim=4,
        group_size=2,
        seed=3,
        device="cpu",
        layer_name="radio_final",
        spatial_samples=0,
    )
    selector = LocalizableFeatureSelector(input_dim=8, output_dim=4, group_size=2)
    device = torch.device("cpu")

    individual = [_score_group(selector, group, config, device) for group in groups]
    batched = _score_groups_batched(selector, groups, config, device)

    assert len(batched) == len(individual)
    for actual, expected in zip(batched, individual):
        torch.testing.assert_close(actual, expected)


def test_utility_weighted_pooling_biases_descriptor_toward_high_utility_position():
    selector = LocalizableFeatureSelector(input_dim=2, output_dim=2, group_size=1)
    with torch.no_grad():
        selector.group_logits.fill_(8.0)
        selector.projection.weight.zero_()
        selector.projection.weight[0, 0, 0, 0] = 1.0
        selector.projection.weight[1, 1, 0, 0] = 1.0
        selector.utility_head.weight.zero_()
        selector.utility_head.weight[0, 0, 0, 0] = 8.0
        selector.utility_head.bias.zero_()

    tokens = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    unweighted = _selected_descriptor(selector, tokens, use_utility_weighted_pooling=False)
    weighted = _selected_descriptor(selector, tokens, use_utility_weighted_pooling=True)

    torch.testing.assert_close(unweighted, torch.tensor([[0.7071, 0.7071]]), rtol=1e-3, atol=1e-3)
    assert float(weighted[0, 0]) > float(weighted[0, 1])


def test_utility_weighted_dense_loss_backpropagates_to_utility_head(tmp_path):
    bank, query_manifest, map_manifest = _synthetic_dense_problem(tmp_path, query_count=3, candidates_per_query=3)
    groups = _build_dense_groups(bank, query_manifest, map_manifest)
    groups = _normalize_dense_groups(groups, _pose_cost_normalizer(groups))
    selector = LocalizableFeatureSelector(input_dim=8, output_dim=4, group_size=2)
    config = DenseSelectorTrainingConfig(
        steps=1,
        batch_size=3,
        output_dim=4,
        group_size=2,
        seed=3,
        device="cpu",
        layer_name="radio_final",
        spatial_samples=0,
        utility_weighted_pooling=True,
    )

    loss = _loss_for_groups(selector, groups, config, torch.device("cpu"))
    loss.backward()

    assert selector.utility_head.weight.grad is not None
    assert float(selector.utility_head.weight.grad.abs().sum()) > 0.0


def test_dense_selector_training_preloads_sampled_features_without_limiting_training_pool(tmp_path, monkeypatch):
    bank, query_manifest, map_manifest = _synthetic_dense_problem(tmp_path, query_count=8, candidates_per_query=3)
    original_read = dense_training._read_dense_feature
    read_count = 0

    def counted_read(path, layer_name):
        nonlocal read_count
        read_count += 1
        return original_read(path, layer_name)

    _load_sampled_dense_feature.cache_clear()
    monkeypatch.setattr(dense_training, "_read_dense_feature", counted_read)

    summary = train_dense_selector(
        bank,
        query_manifest,
        map_manifest,
        DenseSelectorTrainingConfig(
            steps=3,
            batch_size=4,
            output_dim=4,
            group_size=2,
            lr=0.01,
            eval_split_fraction=0.25,
            seed=5,
            device="cpu",
            layer_name="radio_final",
            spatial_samples=4,
            diagnostic_query_limit=2,
            preload_sampled_features=True,
        ),
    )

    assert summary.query_count == 8
    assert summary.train_query_count == 6
    assert summary.preloaded_feature_count > 0
    assert read_count <= summary.preloaded_feature_count + 1


def test_dense_selector_training_writes_and_reuses_persistent_sample_cache(tmp_path, monkeypatch):
    bank, query_manifest, map_manifest = _synthetic_dense_problem(tmp_path, query_count=8, candidates_per_query=3)
    cache_path = tmp_path / "sampled_cache.npz"
    _load_sampled_dense_feature.cache_clear()

    first_summary = train_dense_selector(
        bank,
        query_manifest,
        map_manifest,
        DenseSelectorTrainingConfig(
            steps=2,
            batch_size=4,
            output_dim=4,
            group_size=2,
            lr=0.01,
            eval_split_fraction=0.25,
            seed=5,
            sampling_seed=0,
            device="cpu",
            layer_name="radio_final",
            spatial_samples=4,
            diagnostic_query_limit=2,
            preload_sampled_features=True,
            write_sample_cache=str(cache_path),
        ),
    )
    assert cache_path.exists()
    assert first_summary.preloaded_feature_count > 0
    assert first_summary.written_feature_count == first_summary.preloaded_feature_count

    def fail_read(path, layer_name):
        raise AssertionError(f"unexpected dense NPZ read for {path} {layer_name}")

    _load_sampled_dense_feature.cache_clear()
    monkeypatch.setattr(dense_training, "_read_dense_feature", fail_read)

    second_summary = train_dense_selector(
        bank,
        query_manifest,
        map_manifest,
        DenseSelectorTrainingConfig(
            steps=2,
            batch_size=4,
            output_dim=4,
            group_size=2,
            lr=0.01,
            eval_split_fraction=0.25,
            seed=6,
            sampling_seed=0,
            device="cpu",
            layer_name="radio_final",
            spatial_samples=4,
            diagnostic_query_limit=2,
            sample_cache=str(cache_path),
            preload_sampled_features=True,
        ),
    )

    assert second_summary.loaded_feature_count == first_summary.written_feature_count
    assert second_summary.preloaded_feature_count == first_summary.written_feature_count


def test_sampled_dense_feature_cache_round_trips_keyed_arrays(tmp_path):
    cache_path = tmp_path / "cache.npz"
    cache = {
        ("a.npz", "radio_final", 4, 11): np.ones((2, 4, 1), dtype=np.float32),
        ("b.npz", "radio_final", 4, 12): np.full((2, 4, 1), 2.0, dtype=np.float32),
    }

    _write_sampled_dense_feature_cache(cache, cache_path)
    loaded = _load_sampled_dense_feature_cache(cache_path)

    assert set(loaded) == set(cache)
    for key in cache:
        np.testing.assert_array_equal(loaded[key], cache[key])


def test_build_dense_sample_cache_cli_writes_training_pool_cache(tmp_path):
    bank, query_manifest, map_manifest = _synthetic_dense_problem(tmp_path, query_count=8, candidates_per_query=3)
    bank_path = tmp_path / "candidates.jsonl"
    query_path = tmp_path / "query_manifest.json"
    map_path = tmp_path / "map_manifest.json"
    cache_path = tmp_path / "sample_cache.npz"
    summary_path = tmp_path / "sample_cache_summary.json"
    bank.to_jsonl(bank_path)
    query_manifest.to_json(query_path)
    map_manifest.to_json(map_path)

    build_dense_sample_cache_cli_main(
        [
            "--bank",
            str(bank_path),
            "--query_manifest",
            str(query_path),
            "--map_manifest",
            str(map_path),
            "--layer_name",
            "radio_final",
            "--spatial_samples",
            "4",
            "--seed",
            "5",
            "--sampling_seed",
            "0",
            "--eval_fraction",
            "0.25",
            "--diagnostic_query_limit",
            "2",
            "--output",
            str(cache_path),
            "--summary",
            str(summary_path),
        ]
    )

    loaded = _load_sampled_dense_feature_cache(cache_path)
    summary = json.loads(summary_path.read_text())
    assert len(loaded) == summary["sample_cache_entry_count"]
    assert summary["train_query_count"] == 6
    assert summary["eval_diagnostic_query_count"] == 2


def test_dense_selector_basin_labels_use_absolute_pose_thresholds():
    group = _DenseQueryGroup(
        query_id="query",
        query_path="q.npz",
        candidate_paths=("good.npz", "bad_t.npz", "bad_r.npz"),
        translation_m=np.asarray([0.5, 6.0, 0.5], dtype=np.float32),
        rotation_deg=np.asarray([2.0, 2.0, 15.0], dtype=np.float32),
        costs=np.asarray([0.0, 1.0, 1.0], dtype=np.float32),
    )

    labels = _basin_labels_for_group(
        group,
        translation_threshold_m=5.0,
        rotation_threshold_deg=10.0,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(labels, torch.tensor([[1.0, 0.0, 0.0]]))


def test_dense_selector_hard_negative_margin_penalizes_out_of_basin_top_score():
    labels = torch.tensor([[1.0, 0.0, 0.0]])
    bad_scores = torch.tensor([[0.1, 0.8, 0.2]])
    good_scores = torch.tensor([[0.8, 0.1, 0.2]])

    bad_loss = _hard_negative_margin_loss(bad_scores, labels, margin=0.1)
    good_loss = _hard_negative_margin_loss(good_scores, labels, margin=0.1)

    assert float(bad_loss) > 0.0
    assert float(good_loss) == pytest.approx(0.0)


def test_dense_selector_loss_accepts_basin_and_hard_negative_terms(tmp_path):
    bank, query_manifest, map_manifest = _synthetic_dense_problem(tmp_path, query_count=3, candidates_per_query=3)
    groups = _build_dense_groups(bank, query_manifest, map_manifest)
    groups = _normalize_dense_groups(groups, _pose_cost_normalizer(groups))
    selector = LocalizableFeatureSelector(input_dim=8, output_dim=4, group_size=2)
    base_config = DenseSelectorTrainingConfig(
        steps=1,
        batch_size=3,
        output_dim=4,
        group_size=2,
        seed=3,
        device="cpu",
        layer_name="radio_final",
        spatial_samples=0,
    )
    weighted_config = DenseSelectorTrainingConfig(
        steps=1,
        batch_size=3,
        output_dim=4,
        group_size=2,
        seed=3,
        device="cpu",
        layer_name="radio_final",
        spatial_samples=0,
        basin_bce_weight=0.25,
        hard_negative_weight=0.5,
        basin_translation_threshold_m=0.25,
        basin_rotation_threshold_deg=5.0,
        hard_negative_margin=0.1,
    )

    base_loss = _loss_for_groups(selector, groups, base_config, torch.device("cpu"))
    weighted_loss = _loss_for_groups(selector, groups, weighted_config, torch.device("cpu"))

    assert torch.isfinite(weighted_loss)
    assert float(weighted_loss) > float(base_loss)


def test_dense_selector_joint_track_supervision_requires_track_observations(tmp_path):
    bank, query_manifest, map_manifest = _synthetic_dense_problem(tmp_path, query_count=3, candidates_per_query=3)

    with pytest.raises(ValueError, match="track_observations"):
        run_dense_selector_training(
            bank,
            query_manifest,
            map_manifest,
            DenseSelectorTrainingConfig(
                steps=1,
                batch_size=3,
                output_dim=4,
                group_size=2,
                seed=3,
                device="cpu",
                layer_name="radio_final",
                spatial_samples=0,
                track_supervision_weight=1.0,
            ),
        )


def test_dense_selector_joint_track_supervision_reports_track_diagnostics(tmp_path):
    bank, query_manifest, map_manifest = _synthetic_dense_problem(tmp_path, query_count=8, candidates_per_query=3)

    run = run_dense_selector_training(
        bank,
        query_manifest,
        map_manifest,
        DenseSelectorTrainingConfig(
            steps=60,
            batch_size=4,
            output_dim=4,
            group_size=2,
            lr=0.02,
            eval_split_fraction=0.25,
            seed=7,
            device="cpu",
            layer_name="radio_final",
            spatial_samples=0,
            track_supervision_weight=0.5,
            track_batch_size=4,
            track_contrastive_weight=1.0,
            track_consistency_weight=0.5,
            track_utility_weight=1.0,
        ),
        track_observations=_synthetic_track_observations(track_count=6, channels=8),
    )

    assert run.summary.track_supervision_track_count == 6
    assert run.summary.track_supervision_observation_count == 18
    assert run.summary.final_loss < run.summary.initial_loss
    assert run.summary.final_track_negative_similarity < run.summary.initial_track_negative_similarity
    assert run.summary.final_track_utility_target_correlation > 0.5


def test_dense_selector_init_anchor_limits_warm_start_drift(tmp_path):
    bank, query_manifest, map_manifest = _synthetic_dense_problem(tmp_path, query_count=6, candidates_per_query=3)
    checkpoint_path = tmp_path / "warm_start.pt"
    initial_selector = LocalizableFeatureSelector(input_dim=8, output_dim=4, group_size=2)
    torch.save(initial_selector.state_dict(), checkpoint_path)

    def state_distance(selector):
        initial_state = torch.load(checkpoint_path, map_location="cpu")
        distance = 0.0
        for name, value in selector.state_dict().items():
            distance += float(torch.mean((value.detach().cpu() - initial_state[name]) ** 2))
        return distance

    base_config = DenseSelectorTrainingConfig(
        steps=20,
        batch_size=4,
        output_dim=4,
        group_size=2,
        lr=0.03,
        eval_split_fraction=0.25,
        seed=11,
        device="cpu",
        layer_name="radio_final",
        spatial_samples=0,
        init_checkpoint=str(checkpoint_path),
        track_supervision_weight=1.0,
        track_batch_size=4,
        track_contrastive_weight=1.0,
        track_consistency_weight=1.0,
        track_utility_weight=0.5,
    )
    unanchored = run_dense_selector_training(
        bank,
        query_manifest,
        map_manifest,
        base_config,
        track_observations=_synthetic_track_observations(track_count=6, channels=8),
    )
    anchored = run_dense_selector_training(
        bank,
        query_manifest,
        map_manifest,
        DenseSelectorTrainingConfig(**{**base_config.__dict__, "init_anchor_weight": 10.0}),
        track_observations=_synthetic_track_observations(track_count=6, channels=8),
    )

    assert state_distance(anchored.selector) < state_distance(unanchored.selector) * 0.5


def test_dense_selector_cost_normalizer_uses_train_groups_only():
    train_group = _DenseQueryGroup(
        query_id="train",
        query_path="q_train.npz",
        candidate_paths=("m0.npz", "m1.npz"),
        translation_m=np.asarray([1.0, 2.0], dtype=np.float32),
        rotation_deg=np.asarray([1.0, 2.0], dtype=np.float32),
        costs=np.zeros((2,), dtype=np.float32),
    )
    eval_outlier = _DenseQueryGroup(
        query_id="eval",
        query_path="q_eval.npz",
        candidate_paths=("m2.npz", "m3.npz"),
        translation_m=np.asarray([100.0, 200.0], dtype=np.float32),
        rotation_deg=np.asarray([100.0, 200.0], dtype=np.float32),
        costs=np.zeros((2,), dtype=np.float32),
    )

    normalizer = _pose_cost_normalizer([train_group])
    normalized_train = _normalize_dense_groups([train_group], normalizer)[0]
    normalized_eval = _normalize_dense_groups([eval_outlier], normalizer)[0]

    assert normalizer == pytest.approx((2.0, 2.0))
    assert normalized_train.costs.tolist() == pytest.approx([0.5, 1.0])
    assert normalized_eval.costs.tolist() == pytest.approx([50.0, 100.0])


def test_dense_selector_training_rejects_missing_reference_tokens(tmp_path):
    bank, query_manifest, map_manifest = _synthetic_dense_problem(tmp_path, query_count=1)
    bad_map_manifest = TokenBankManifest(records=map_manifest.records[:1])

    with pytest.raises(ValueError, match="reference token not found"):
        train_dense_selector(
            bank,
            query_manifest,
            bad_map_manifest,
            DenseSelectorTrainingConfig(steps=1, output_dim=4, group_size=2),
        )
