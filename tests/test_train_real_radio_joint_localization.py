from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from feature_extract.tools.vfm import train_real_radio_joint_localization
from feature_extract.vfm.matcha_coarse_fine_adapter import MatchaCoarseFineTrainingSet
from feature_extract.vfm.matcha_joint_training import MatchaJointTrainingSet


def _coarse_rows() -> MatchaCoarseFineTrainingSet:
    return MatchaCoarseFineTrainingSet(
        query_features=np.ones((1, 4), dtype=np.float32),
        render_features=np.ones((1, 4), dtype=np.float32),
        query_offset_labels=np.asarray([0], dtype=np.int64),
        render_offset_labels=np.asarray([0], dtype=np.int64),
        negative_render_features=np.zeros((1, 1, 4), dtype=np.float32),
        roundtrip_errors_px=np.zeros((1,), dtype=np.float32),
    )


def _full_joint_set() -> MatchaJointTrainingSet:
    return MatchaJointTrainingSet(
        coarse_fine_samples=_coarse_rows(),
        query_feature_maps=np.ones((1, 4, 2, 2), dtype=np.float32),
        render_feature_maps=np.ones((1, 4, 2, 2), dtype=np.float32),
        query_heatmap_targets=np.zeros((1, 2, 2), dtype=np.float32),
        render_heatmap_targets=np.zeros((1, 2, 2), dtype=np.float32),
        sample_pair_indices=np.asarray([0], dtype=np.int64),
        query_cell_indices=np.asarray([0], dtype=np.int64),
        render_cell_indices=np.asarray([0], dtype=np.int64),
        sample_track_ids=np.asarray([7], dtype=np.int64),
        sample_track_xyz=np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64),
        landmark_sample_pair_indices=np.asarray([0], dtype=np.int64),
        landmark_query_xy=np.asarray([[4.0, 4.0]], dtype=np.float64),
        landmark_reference_xy=np.asarray([[4.0, 4.0]], dtype=np.float64),
        landmark_track_ids=np.asarray([7], dtype=np.int64),
        landmark_track_xyz=np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64),
        landmark_support_view_counts=np.asarray([2], dtype=np.int64),
        pair_query_image_sizes=np.asarray([[16, 16]], dtype=np.int64),
        pair_reference_image_sizes=np.asarray([[16, 16]], dtype=np.int64),
        fine_sample_pair_indices=np.asarray([0], dtype=np.int64),
        fine_query_cell_indices=np.asarray([0], dtype=np.int64),
        fine_render_cell_indices=np.asarray([0], dtype=np.int64),
        fine_query_offset_labels=np.asarray([0], dtype=np.int64),
        fine_render_offset_labels=np.asarray([0], dtype=np.int64),
        fine_query_xy=np.asarray([[4.0, 4.0]], dtype=np.float32),
        fine_render_xy=np.asarray([[4.0, 4.0]], dtype=np.float32),
        query_rgb_images=np.zeros((1, 3, 16, 16), dtype=np.float32),
        render_rgb_images=np.zeros((1, 3, 16, 16), dtype=np.float32),
    )


def test_train_real_radio_joint_localization_cli_defaults_to_full_joint_training() -> None:
    args = train_real_radio_joint_localization.parse_args(
        [
            "--joint_cache",
            "train.npz",
            "--output_model",
            "adapter.pt",
            "--output_joint_model",
            "joint.pt",
            "--summary_json",
            "summary.json",
            "--steps",
            "2",
            "--device",
            "cpu",
        ]
    )

    assert args.joint_cache == "train.npz"
    assert args.model_type == "residual_adapter"
    assert args.measurement_patch_loss_weight == pytest.approx(1.0)
    assert args.landmark_retrieval_loss_weight == pytest.approx(0.25)
    assert args.local_window_fine_loss_weight > 0.0
    assert not hasattr(args, "sample_cache")
    assert not hasattr(args, "render_cache_manifest_csv")
    assert args.landmark_episode_support_pairs == 0
    assert args.landmark_prototype_aggregation_method == "mean"
    assert args.landmark_l2_normalize_observations is False
    assert args.landmark_normalize_final_prototypes is True
    assert args.landmark_frozen_negative_bank == ""
    assert args.landmark_frozen_bank_support_observations == ""
    assert train_real_radio_joint_localization._requires_rgb_training(args) is True


def test_retrieval_only_training_does_not_require_rgb() -> None:
    args = train_real_radio_joint_localization.parse_args(
        [
            "--joint_cache",
            "train.npz",
            "--output_model",
            "adapter.pt",
            "--output_joint_model",
            "joint.pt",
            "--summary_json",
            "summary.json",
            "--measurement_patch_loss_weight",
            "0",
            "--rgb_keypoint_loss_weight",
            "0",
            "--rgb_keypoint_position_loss_weight",
            "0",
        ]
    )

    assert train_real_radio_joint_localization._requires_rgb_training(args) is False

    audit = train_real_radio_joint_localization._validate_joint_localization_training_set(
        replace(_full_joint_set(), query_rgb_images=None, render_rgb_images=None),
        source="train",
        require_landmark_retrieval_supervision=True,
    )
    assert audit["query_rgb_shape"] is None
    assert audit["reference_rgb_shape"] is None


@pytest.mark.parametrize(
    ("rank", "expected_events"),
    ((0, ["load", "barrier"]), (1, ["barrier", "load"])),
)
def test_distributed_track_index_cache_is_built_by_rank_zero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    rank: int,
    expected_events: list[str],
) -> None:
    events: list[str] = []
    sentinel = object()
    args = SimpleNamespace(
        landmark_track_observations=str(tmp_path / "tracks.jsonl"),
        landmark_track_observation_index_cache=str(tmp_path / "tracks.index.npz"),
    )
    monkeypatch.setattr(
        train_real_radio_joint_localization,
        "load_track_observation_index",
        lambda *_args, **_kwargs: events.append("load") or sentinel,
    )
    monkeypatch.setattr(
        train_real_radio_joint_localization.torch.distributed,
        "barrier",
        lambda: events.append("barrier"),
    )

    result = train_real_radio_joint_localization._load_training_track_observation_index(
        args,
        {"enabled": True, "rank": rank},
    )

    assert result is sentinel
    assert events == expected_events


def test_multiview_episode_provider_groups_distinct_support_images() -> None:
    records = [
        {"query_id": "q0", "reference_image_id": "r0"},
        {"query_id": "q0", "reference_image_id": "r1"},
        {"query_id": "q0", "reference_image_id": "r2"},
        {"query_id": "q1", "reference_image_id": "r3"},
        {"query_id": "q1", "reference_image_id": "r4"},
    ]

    class BaseProvider:
        metadata = {"records": records}

        def get(self, index):
            record = records[int(index)]
            return replace(
                _full_joint_set(),
                pair_query_ids=np.asarray([record["query_id"]], dtype=object),
                pair_candidate_ids=np.asarray([record["reference_image_id"]], dtype=object),
            )

        def landmark_retrieval_audit(self):
            return {"supervision_source": "sfm_common_track_observations", "unique_track_count": 1}

    provider = train_real_radio_joint_localization.RealRadioMultiViewEpisodeProvider(
        BaseProvider(),
        support_pairs=2,
        min_support_pairs=2,
        seed=3,
    )
    episode = provider.get(0)

    assert len(provider) == 3
    assert episode.query_feature_maps.shape[0] == 2
    assert len(set(episode.pair_candidate_ids.tolist())) == 2
    assert len(set(episode.pair_query_ids.tolist())) == 1
    assert provider.landmark_retrieval_audit()["query_excluded_from_support"] is True


def test_track_centric_episode_keeps_target_in_every_support_view() -> None:
    records = [{"query_id": "q0", "reference_image_id": "r0"}]
    tracks_by_image = {
        "q0": [7, 8],
        "r0": [7],
        "r1": [7, 8],
        "r2": [7],
    }

    class BaseProvider:
        metadata = {"records": records}
        track_observation_index = SimpleNamespace(
            by_image={
                image_id: SimpleNamespace(track_ids=np.asarray(track_ids, dtype=np.int64))
                for image_id, track_ids in tracks_by_image.items()
            }
        )

        def get_sfm_pair(self, query_id, reference_id):
            common = sorted(set(tracks_by_image[query_id]).intersection(tracks_by_image[reference_id]))
            count = len(common)
            return replace(
                _full_joint_set(),
                pair_query_ids=np.asarray([query_id], dtype=object),
                pair_candidate_ids=np.asarray([reference_id], dtype=object),
                landmark_sample_pair_indices=np.zeros((count,), dtype=np.int64),
                landmark_query_xy=np.tile(np.asarray([[4.0, 4.0]]), (count, 1)),
                landmark_reference_xy=np.tile(np.asarray([[4.0, 4.0]]), (count, 1)),
                landmark_track_ids=np.asarray(common, dtype=np.int64),
                landmark_track_xyz=np.zeros((count, 3), dtype=np.float64),
                landmark_support_view_counts=np.full((count,), 3, dtype=np.int64),
            )

        def landmark_retrieval_audit(self):
            return {"supervision_source": "sfm_common_track_observations", "unique_track_count": 2}

    provider = train_real_radio_joint_localization.RealRadioMultiViewEpisodeProvider(
        BaseProvider(),
        support_pairs=3,
        min_support_pairs=2,
        seed=5,
        support_selection="sfm_track_episode",
        episodes_per_query=1,
        allowed_support_image_ids={"r0", "r1", "r2"},
    )
    episode = provider.get(0)
    audit = provider.landmark_retrieval_audit()

    support_count = int(episode.query_feature_maps.shape[0])
    assert 2 <= support_count <= 3
    assert np.count_nonzero(episode.landmark_track_ids == 7) == support_count
    assert audit["episode_target_track_count"] == 1
    assert audit["episode_target_track_support_count_mean"] == pytest.approx(float(support_count))


def test_frozen_landmark_bank_contract_rejects_stale_support_split(tmp_path: Path) -> None:
    checkpoint = tmp_path / "joint.pt"
    checkpoint.write_text("checkpoint")
    support = tmp_path / "support.jsonl"
    support.write_text("support\n")
    from feature_extract.vfm.artifacts import file_sha256_short

    bank = tmp_path / "bank.npz"
    metadata = {
        "descriptor_space_id": "space",
        "descriptor_space_manifest": {"version": 2, "projection_source": "projected_observation_full_map"},
        "track_observations_sha256": file_sha256_short(support),
        "matcha_joint_checkpoint_sha256": file_sha256_short(checkpoint),
        "source_image_count": 3,
    }
    np.savez(
        bank,
        track_ids=np.asarray([1], dtype=np.int64),
        features=np.ones((1, 4), dtype=np.float32),
        metadata_json=np.asarray(json.dumps(metadata)),
    )

    audit = train_real_radio_joint_localization.validate_frozen_landmark_bank_contract(
        bank,
        warm_start_checkpoint=checkpoint,
        support_observations=support,
        expected_source_image_count=3,
    )
    stale_support = tmp_path / "stale.jsonl"
    stale_support.write_text("different\n")

    assert audit["validated"] is True
    with pytest.raises(ValueError, match="support-observation mismatch"):
        train_real_radio_joint_localization.validate_frozen_landmark_bank_contract(
            bank,
            warm_start_checkpoint=checkpoint,
            support_observations=stale_support,
            expected_source_image_count=3,
        )


def test_upstream_disjoint_manifest_contract_rejects_either_pair_side(tmp_path: Path) -> None:
    from feature_extract.vfm.artifacts import file_sha256_short

    query_split = tmp_path / "query_split.json"
    query_split.write_text(json.dumps({"train": ["q.png"], "validation": [], "test": []}))
    manifest = tmp_path / "manifest.json"
    base = {
        "heldout_image_filter": {
            "query_split_sha256": file_sha256_short(query_split),
            "heldout_image_count": 1,
        }
    }
    manifest.write_text(json.dumps({**base, "records": [{"query_id": "a.png", "reference_image_id": "b.png"}]}))

    audit = train_real_radio_joint_localization.validate_upstream_disjoint_manifest_contract(
        manifest,
        query_split_path=query_split,
    )

    assert audit["validated"] is True
    assert audit["leaked_pair_count"] == 0
    for record in (
        {"query_id": "q.png", "reference_image_id": "b.png"},
        {"query_id": "a.png", "reference_image_id": "q.png"},
    ):
        manifest.write_text(json.dumps({**base, "records": [record]}))
        with pytest.raises(ValueError, match="held-out query images"):
            train_real_radio_joint_localization.validate_upstream_disjoint_manifest_contract(
                manifest,
                query_split_path=query_split,
            )


def test_episode_support_manifest_uses_full_disjoint_scope(tmp_path: Path) -> None:
    token_path = tmp_path / "token.npz"
    np.savez(token_path, radio_final=np.ones((4, 2, 2), dtype=np.float32))
    query_split = tmp_path / "query_split.json"
    query_split.write_text(json.dumps({"train": ["q.png"], "validation": [], "test": []}))

    def record(image_id: str) -> dict[str, object]:
        return {
            "image_id": image_id,
            "token_path": str(token_path),
            "layers": [
                {
                    "name": "radio_final",
                    "model": "C-RADIO",
                    "layer": "final",
                    "channels": 4,
                    "stride": 16,
                }
            ],
            "split": "train",
            "scene": "scene",
        }

    support_manifest = tmp_path / "support.json"
    support_manifest.write_text(json.dumps({"records": [record("a.png"), record("b.png")]}))
    image_ids, audit = train_real_radio_joint_localization.load_episode_support_image_ids(
        support_manifest,
        upstream_disjoint_query_split=query_split,
    )

    assert image_ids == {"a.png", "b.png"}
    assert audit["image_count"] == 2
    support_manifest.write_text(json.dumps({"records": [record("a.png"), record("q.png")]}))
    with pytest.raises(ValueError, match="held-out query images"):
        train_real_radio_joint_localization.load_episode_support_image_ids(
            support_manifest,
            upstream_disjoint_query_split=query_split,
        )


def test_validate_joint_localization_training_set_rejects_row_only_samples() -> None:
    row_only = MatchaJointTrainingSet(coarse_fine_samples=_coarse_rows())

    with pytest.raises(ValueError, match="full-map"):
        train_real_radio_joint_localization._validate_joint_localization_training_set(row_only, source="train")


def test_validate_joint_localization_training_set_requires_fine_supervision_for_measurement_loss() -> None:
    samples = MatchaJointTrainingSet(
        coarse_fine_samples=_coarse_rows(),
        query_feature_maps=np.ones((1, 4, 2, 2), dtype=np.float32),
        render_feature_maps=np.ones((1, 4, 2, 2), dtype=np.float32),
        query_heatmap_targets=np.zeros((1, 2, 2), dtype=np.float32),
        render_heatmap_targets=np.zeros((1, 2, 2), dtype=np.float32),
        sample_pair_indices=np.asarray([0], dtype=np.int64),
        query_cell_indices=np.asarray([0], dtype=np.int64),
        render_cell_indices=np.asarray([0], dtype=np.int64),
        query_rgb_images=np.zeros((1, 3, 16, 16), dtype=np.float32),
        render_rgb_images=np.zeros((1, 3, 16, 16), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="fine.*measurement"):
        train_real_radio_joint_localization._validate_joint_localization_training_set(
            samples,
            source="train",
            require_measurement_supervision=True,
        )


def test_validate_joint_localization_training_set_allows_different_query_reference_grid_sizes() -> None:
    samples = MatchaJointTrainingSet(
        coarse_fine_samples=_coarse_rows(),
        query_feature_maps=np.ones((1, 4, 2, 2), dtype=np.float32),
        render_feature_maps=np.ones((1, 4, 1, 2), dtype=np.float32),
        query_heatmap_targets=np.zeros((1, 2, 2), dtype=np.float32),
        render_heatmap_targets=np.zeros((1, 1, 2), dtype=np.float32),
        sample_pair_indices=np.asarray([0], dtype=np.int64),
        query_cell_indices=np.asarray([0], dtype=np.int64),
        render_cell_indices=np.asarray([0], dtype=np.int64),
        fine_sample_pair_indices=np.asarray([0], dtype=np.int64),
        fine_query_cell_indices=np.asarray([0], dtype=np.int64),
        fine_render_cell_indices=np.asarray([0], dtype=np.int64),
        fine_query_offset_labels=np.asarray([0], dtype=np.int64),
        fine_render_offset_labels=np.asarray([0], dtype=np.int64),
        fine_query_xy=np.asarray([[4.0, 4.0]], dtype=np.float32),
        fine_render_xy=np.asarray([[4.0, 4.0]], dtype=np.float32),
        query_rgb_images=np.zeros((1, 3, 16, 16), dtype=np.float32),
        render_rgb_images=np.zeros((1, 3, 8, 16), dtype=np.float32),
    )

    audit = train_real_radio_joint_localization._validate_joint_localization_training_set(
        samples,
        source="train",
        require_measurement_supervision=True,
    )

    assert audit["query_feature_map_shape"] == [1, 4, 2, 2]
    assert audit["reference_feature_map_shape"] == [1, 4, 1, 2]


def test_train_real_radio_joint_localization_main_uses_joint_backend(monkeypatch, tmp_path: Path) -> None:
    captured = {}
    full_set = _full_joint_set()

    def fake_load_npz(path):
        captured["loaded_path"] = Path(path)
        return full_set, {"format": "fake_real_joint_cache"}

    def fake_train(samples, config, **kwargs):
        captured["samples"] = samples
        captured["config"] = config
        captured["train_kwargs"] = kwargs
        return SimpleNamespace(model=SimpleNamespace(), summary={"sample_count": 1, "final_loss": 0.0})

    monkeypatch.setattr(train_real_radio_joint_localization, "load_matcha_joint_training_set_npz", fake_load_npz)
    monkeypatch.setattr(train_real_radio_joint_localization, "train_matcha_joint_model", fake_train)
    monkeypatch.setattr(train_real_radio_joint_localization, "joint_run_as_coarse_fine_adapter_run", lambda run: run)
    monkeypatch.setattr(train_real_radio_joint_localization, "save_matcha_coarse_fine_adapter", lambda run, path: captured.setdefault("adapter_path", Path(path)))
    monkeypatch.setattr(train_real_radio_joint_localization, "save_matcha_joint_model", lambda run, path: captured.setdefault("joint_path", Path(path)))

    summary_json = tmp_path / "summary.json"
    train_real_radio_joint_localization.main(
        [
            "--joint_cache",
            "train.npz",
            "--output_model",
            str(tmp_path / "adapter.pt"),
            "--output_joint_model",
            str(tmp_path / "joint.pt"),
            "--summary_json",
            str(summary_json),
            "--steps",
            "3",
            "--device",
            "cpu",
        ]
    )

    assert captured["loaded_path"] == Path("train.npz")
    assert captured["samples"] is full_set
    assert captured["config"].model_type == "residual_adapter"
    assert captured["config"].measurement_patch_loss_weight == pytest.approx(1.0)
    assert captured["config"].landmark_retrieval_loss_weight == pytest.approx(0.25)
    assert captured["config"].steps == 3
    assert captured["adapter_path"] == tmp_path / "adapter.pt"
    assert captured["joint_path"] == tmp_path / "joint.pt"
    summary = json.loads(summary_json.read_text())
    assert summary["stage"] == "real_radio_joint_localization_training"
    assert summary["joint_training_contract"]["requires_full_feature_maps"] is True
    assert summary["joint_training_contract"]["requires_rgb_measurement_images"] is True


def test_train_real_radio_joint_localization_main_uses_referenced_lazy_provider(monkeypatch, tmp_path: Path) -> None:
    captured = {}
    full_set = _full_joint_set()
    manifest = tmp_path / "referenced_manifest.json"
    manifest.write_text(json.dumps({"format": "vfm_real_radio_joint_referenced_manifest_v1", "records": [{}]}) + "\n")

    class FakeProvider:
        def __init__(self, path, **kwargs):
            captured.setdefault("provider_paths", []).append(Path(path))
            captured.setdefault("provider_kwargs", []).append(dict(kwargs))

        def __len__(self):
            return 3

        def get(self, index):
            captured.setdefault("provider_get_indices", []).append(int(index))
            return full_set

        def landmark_retrieval_audit(self):
            return {"unique_track_count": 1, "supervision_source": "sfm_common_track_observations"}

    def fake_train_from_provider(sample_count, get_sample, config, **kwargs):
        captured["provider_sample_count"] = int(sample_count)
        captured["provider_callable"] = get_sample
        captured["config"] = config
        captured["train_kwargs"] = kwargs
        assert get_sample(1) is full_set
        return SimpleNamespace(model=SimpleNamespace(), summary={"sample_count": 3, "provider_lazy_training": True})

    monkeypatch.setattr(train_real_radio_joint_localization, "RealRadioReferencedJointSampleProvider", FakeProvider)
    monkeypatch.setattr(train_real_radio_joint_localization, "train_matcha_joint_model_from_sample_provider", fake_train_from_provider)
    monkeypatch.setattr(train_real_radio_joint_localization, "joint_run_as_coarse_fine_adapter_run", lambda run: run)
    monkeypatch.setattr(train_real_radio_joint_localization, "save_matcha_coarse_fine_adapter", lambda run, path: captured.setdefault("adapter_path", Path(path)))
    monkeypatch.setattr(train_real_radio_joint_localization, "save_matcha_joint_model", lambda run, path: captured.setdefault("joint_path", Path(path)))

    summary_json = tmp_path / "summary.json"
    train_real_radio_joint_localization.main(
        [
            "--joint_cache_manifest",
            str(manifest),
            "--output_model",
            str(tmp_path / "adapter.pt"),
            "--output_joint_model",
            str(tmp_path / "joint.pt"),
            "--summary_json",
            str(summary_json),
            "--steps",
            "2",
            "--referenced_feature_cache_size",
            "256",
            "--referenced_rgb_cache_size",
            "128",
            "--provider_prefetch_workers",
            "2",
            "--provider_prefetch_depth",
            "8",
            "--provider_gradient_accumulation_pairs",
            "4",
            "--provider_pair_batch_size",
            "2",
            "--provider_progress_interval_steps",
            "25",
            "--device",
            "cpu",
        ]
    )

    assert captured["provider_paths"] == [manifest]
    assert captured["provider_kwargs"] == [
        {"feature_cache_size": 256, "rgb_cache_size": 128, "load_rgb": True}
    ]
    assert captured["provider_get_indices"] == [0, 1]
    assert captured["provider_sample_count"] == 3
    assert captured["config"].model_type == "residual_adapter"
    assert captured["train_kwargs"]["provider_name"] == "real_radio_referenced_manifest"
    assert captured["train_kwargs"]["provider_prefetch_workers"] == 2
    assert captured["train_kwargs"]["provider_prefetch_depth"] == 8
    assert captured["train_kwargs"]["provider_gradient_accumulation_pairs"] == 4
    assert captured["train_kwargs"]["provider_pair_batch_size"] == 2
    assert captured["train_kwargs"]["provider_progress_interval_steps"] == 25
    assert captured["adapter_path"] == tmp_path / "adapter.pt"
    assert captured["joint_path"] == tmp_path / "joint.pt"
    summary = json.loads(summary_json.read_text())
    assert summary["joint_cache"]["metadata"]["format"] == "vfm_real_radio_joint_referenced_manifest_v1"
    assert "records" not in summary["joint_cache"]["metadata"]
    assert summary["joint_cache"]["metadata"]["record_count"] == 1
    assert summary["training"]["provider_lazy_training"] is True
