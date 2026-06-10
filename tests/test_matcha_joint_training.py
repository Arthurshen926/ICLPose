from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch

import feature_extract.vfm.matcha_joint_training as joint_training
from feature_extract.extractors.extractor_radio import RADIOFeatureExtractor
from feature_extract.tools.vfm.build_matcha_joint_cache import (
    _extract_matcha_joint_feature_from_rgb,
    _render_pair_cache_key,
    _write_matcha_joint_cache_outputs,
    parse_args as parse_build_matcha_joint_cache_args,
)
from feature_extract.tools.vfm.train_matcha_joint_model import main as train_matcha_joint_model_cli
from feature_extract.tools.vfm.train_matcha_joint_streaming_model import parse_args as parse_streaming_train_args
from feature_extract.vfm.matcha_coarse_fine_adapter import MatchaCoarseFineTrainingSet
from feature_extract.vfm.matcha_coarse_supervision import MatchaCoarseSupervision
from feature_extract.vfm.matcha_joint_cache import (
    _mine_negative_render_indices,
    build_matcha_joint_index_training_set_from_maps,
    build_matcha_joint_training_set_from_maps,
    heatmap_targets_from_coarse_supervision,
)
from feature_extract.vfm.matcha_joint_training import (
    MatchaJointTrainingConfig,
    MatchaJointTrainingSet,
    RadioDualAttentionFusionJointModel,
    MatchaStyleJointModel,
    load_matcha_joint_model,
    load_matcha_joint_training_set_manifest,
    load_matcha_joint_training_set_npz,
    merge_matcha_joint_training_sets,
    project_feature_map_with_matcha_joint_model,
    save_matcha_joint_model,
    save_matcha_joint_training_set_manifest,
    save_matcha_joint_training_set_npz,
    train_matcha_joint_model,
    train_matcha_joint_model_from_manifest,
)


def test_render_pair_cache_key_includes_actual_perturbation_metadata() -> None:
    common_kwargs = {
        "pair_type": "B_trans025",
        "render_pose_world_offset": "0,0,0",
        "pair_type_B_translation_m": 0.25,
        "pair_type_C_translation_m": 0.50,
        "perturb_rotation_deg": 2.0,
        "seed": 7,
        "record_index": 3,
        "pair_index": 1,
    }
    key_a = _render_pair_cache_key(
        pair_metadata={"perturb_translation_m": 0.12, "perturb_rotation_deg": 0.4, "candidate_id": ""},
        **common_kwargs,
    )
    key_b = _render_pair_cache_key(
        pair_metadata={"perturb_translation_m": 0.21, "perturb_rotation_deg": 0.4, "candidate_id": ""},
        **common_kwargs,
    )
    key_c = _render_pair_cache_key(
        pair_metadata={"perturb_translation_m": 0.12, "perturb_rotation_deg": 0.4, "candidate_id": "ref/image:01"},
        **common_kwargs,
    )

    assert key_a != key_b
    assert key_a != key_c
    assert "/" not in key_c
    assert ":" not in key_c
    assert "," not in key_c
from feature_extract.vfm.matcha_rgb_keypoint_detector import matcha_keypoint_position_loss


def _toy_samples() -> MatchaCoarseFineTrainingSet:
    features = np.eye(8, dtype=np.float32)
    return MatchaCoarseFineTrainingSet(
        query_features=features,
        render_features=features,
        query_offset_labels=np.arange(8, dtype=np.int64),
        render_offset_labels=np.arange(8, dtype=np.int64),
        negative_render_features=np.roll(features, shift=1, axis=0)[:, None, :],
        roundtrip_errors_px=np.zeros((8,), dtype=np.float32),
        metadata={"toy": True},
    )


def _toy_supervision() -> MatchaCoarseSupervision:
    return MatchaCoarseSupervision(
        query_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
        render_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
        query_xy=np.zeros((4, 2), dtype=np.float64),
        render_xy=np.zeros((4, 2), dtype=np.float64),
        query_offset_labels=np.asarray([0, 9, 18, 36], dtype=np.int64),
        render_offset_labels=np.asarray([36, 18, 9, 0], dtype=np.int64),
        roundtrip_errors_px=np.asarray([0.0, 0.5, 1.0, 3.0], dtype=np.float32),
    )


def _toy_robust_supervision() -> MatchaCoarseSupervision:
    soft = np.zeros((2, 65), dtype=np.float32)
    soft[:, 36] = 0.8
    soft[:, 37] = 0.2
    return MatchaCoarseSupervision(
        query_indices=np.asarray([0, 1], dtype=np.int64),
        render_indices=np.asarray([0, 1], dtype=np.int64),
        query_xy=np.zeros((2, 2), dtype=np.float64),
        render_xy=np.zeros((2, 2), dtype=np.float64),
        query_offset_labels=np.asarray([36, 36], dtype=np.int64),
        render_offset_labels=np.asarray([36, 36], dtype=np.int64),
        roundtrip_errors_px=np.asarray([0.0, 0.25], dtype=np.float32),
        query_offset_soft_labels=soft,
        render_offset_soft_labels=soft,
        confidence_targets=np.asarray([1.0, 0.35], dtype=np.float32),
        confidence_ignore_mask=np.asarray([False, True], dtype=bool),
        uncertainty_px=np.asarray([0.5, 4.0], dtype=np.float32),
        no_match_query_indices=np.asarray([2], dtype=np.int64),
        no_match_render_indices=np.asarray([3], dtype=np.int64),
        no_match_query_offset_labels=np.asarray([64], dtype=np.int64),
        no_match_render_offset_labels=np.asarray([64], dtype=np.int64),
        no_match_roundtrip_errors_px=np.asarray([np.inf], dtype=np.float32),
        no_match_reason_ids=np.asarray([3], dtype=np.int64),
        no_match_confidence_ignore_mask=np.asarray([True], dtype=bool),
    )


def test_negative_mining_excludes_patch_level_alternative_positives() -> None:
    query = np.asarray([[1.0, 0.0]], dtype=np.float32)
    render = np.asarray(
        [
            [0.90, 0.10],
            [1.00, 0.00],
            [0.00, 1.00],
        ],
        dtype=np.float32,
    )

    negatives = _mine_negative_render_indices(
        query,
        render,
        np.asarray([0], dtype=np.int64),
        count=1,
        excluded_render_indices_by_query=(np.asarray([0, 1], dtype=np.int64),),
    )

    assert negatives.tolist() == [[2]]


class _FakeRadioExtractor:
    def extract(self, _tensor):
        return {
            "local": torch.ones(4, 2, 2),
            "summary": torch.zeros(1),
        }

    def extract_dual(self, _tensor, **_kwargs):
        return {
            "fine": torch.ones(4, 2, 2),
            "coarse": torch.full((4, 2, 2), 2.0),
            "dual": torch.cat([torch.ones(4, 2, 2), torch.full((4, 2, 2), 2.0)], dim=0),
        }


class _ConfidenceOnlyModel:
    def forward_rows(self, features):
        return features, torch.zeros((features.shape[0], 65), dtype=features.dtype, device=features.device)

    def encode(self, features):
        return features

    def pair_confidence_logits(self, query_descriptors, render_descriptors):
        return query_descriptors[:, 0]

    def pair_fine_logits(self, query_descriptors, render_descriptors):
        return torch.zeros((query_descriptors.shape[0], 64), dtype=query_descriptors.dtype, device=query_descriptors.device)

    def query_pair_fine_logits(self, query_descriptors, render_descriptors):
        return torch.zeros((query_descriptors.shape[0], 64), dtype=query_descriptors.dtype, device=query_descriptors.device)


class _CountingRadioModel:
    patch_size = 16

    def __init__(self) -> None:
        self.forward_intermediate_calls: list[dict[str, object]] = []
        self.final_calls = 0

    def get_nearest_supported_resolution(self, height: int, width: int):
        return SimpleNamespace(height=int(height), width=int(width))

    def forward_intermediates(self, image, **kwargs):
        self.forward_intermediate_calls.append(dict(kwargs))
        fine = torch.ones(1, 4, 2, 2)
        coarse = torch.full((1, 4, 2, 2), 2.0)
        indices = list(kwargs.get("indices") or [])
        if kwargs.get("intermediates_only", False):
            if len(indices) >= 2:
                return [fine, coarse]
            return [fine]
        return SimpleNamespace(summary=torch.zeros(1, 1), features=coarse), [fine]

    def __call__(self, image, **kwargs):
        self.final_calls += 1
        return torch.zeros(1, 1), torch.full((1, 4, 2, 2), 2.0)


def _fake_radio_feature_extractor(model: _CountingRadioModel) -> RADIOFeatureExtractor:
    extractor = RADIOFeatureExtractor.__new__(RADIOFeatureExtractor)
    extractor.device = torch.device("cpu")
    extractor.model = model
    extractor.patch_size = 16
    extractor.radio_repo = ""
    return extractor


def test_radio_extract_dual_uses_single_forward_when_coarse_is_final() -> None:
    model = _CountingRadioModel()
    extractor = _fake_radio_feature_extractor(model)
    image = torch.zeros(1, 3, 32, 32)

    output = extractor.extract_dual(image, coarse_source="final")

    assert output["dual"].shape == (8, 2, 2)
    assert np.allclose(output["fine"].numpy(), 1.0)
    assert np.allclose(output["coarse"].numpy(), 2.0)
    assert len(model.forward_intermediate_calls) == 1
    assert model.forward_intermediate_calls[0]["intermediates_only"] is False
    assert model.final_calls == 0


def test_radio_extract_dual_uses_single_forward_for_two_intermediate_layers() -> None:
    model = _CountingRadioModel()
    extractor = _fake_radio_feature_extractor(model)
    image = torch.zeros(1, 3, 32, 32)

    output = extractor.extract_dual(
        image,
        fine_intermediate_index=-6,
        coarse_source="intermediate",
        coarse_intermediate_index=-1,
    )

    assert output["dual"].shape == (8, 2, 2)
    assert np.allclose(output["fine"].numpy(), 1.0)
    assert np.allclose(output["coarse"].numpy(), 2.0)
    assert len(model.forward_intermediate_calls) == 1
    assert model.forward_intermediate_calls[0]["indices"] == [-6, -1]
    assert model.final_calls == 0


def test_build_matcha_joint_cache_radio_dual_defaults_to_radio_dual_layer() -> None:
    args = parse_build_matcha_joint_cache_args(
        [
            "--query_manifest",
            "queries.json",
            "--query_pose_file",
            "poses.txt",
            "--image_root",
            "images",
            "--gaussian_rgb_ply",
            "scene.ply",
            "--output",
            "cache.npz",
            "--summary_json",
            "summary.json",
            "--feature_mode",
            "radio_dual",
        ]
    )

    assert args.layer_name == "radio_dual"


def test_build_matcha_joint_cache_accepts_perturbation_pair_types() -> None:
    args = parse_build_matcha_joint_cache_args(
        [
            "--query_manifest",
            "queries.json",
            "--query_pose_file",
            "poses.txt",
            "--image_root",
            "images",
            "--gaussian_rgb_ply",
            "scene.ply",
            "--output",
            "cache.npz",
            "--summary_json",
            "summary.json",
            "--pair_types",
            "A_gt,B_trans025,C_trans050",
            "--pair_type_B_translation_m",
            "0.25",
            "--pair_type_C_translation_m",
            "0.5",
            "--perturb_rotation_deg",
            "7.5",
        ]
    )

    assert args.pair_types == "A_gt,B_trans025,C_trans050"
    assert args.pair_type_B_translation_m == 0.25
    assert args.pair_type_C_translation_m == 0.5
    assert args.perturb_rotation_deg == 7.5


def test_streaming_train_render_pair_fine_loss_alias_overrides_render_loss() -> None:
    args = parse_streaming_train_args(
        [
            "--streaming_manifest",
            "streaming.json",
            "--query_pose_file",
            "poses.txt",
            "--image_root",
            "images",
            "--gaussian_rgb_ply",
            "scene.ply",
            "--output_model",
            "adapter.pt",
            "--summary_json",
            "summary.json",
            "--pair_fine_loss_weight",
            "0.0",
            "--render_pair_fine_loss_weight",
            "0.75",
        ]
    )

    assert args.pair_fine_loss_weight == 0.75


def test_build_matcha_joint_cache_exposes_index_only_format() -> None:
    default_args = parse_build_matcha_joint_cache_args(
        [
            "--query_manifest",
            "queries.json",
            "--query_pose_file",
            "poses.txt",
            "--image_root",
            "images",
            "--gaussian_rgb_ply",
            "scene.ply",
            "--output",
            "cache.npz",
            "--summary_json",
            "summary.json",
        ]
    )
    index_args = parse_build_matcha_joint_cache_args(
        [
            "--query_manifest",
            "queries.json",
            "--query_pose_file",
            "poses.txt",
            "--image_root",
            "images",
            "--gaussian_rgb_ply",
            "scene.ply",
            "--output",
            "cache.npz",
            "--summary_json",
            "summary.json",
            "--cache_format",
            "index_v2",
        ]
    )

    assert default_args.cache_format == "dense"
    assert index_args.cache_format == "index_v2"


def test_build_matcha_joint_index_training_set_preserves_robust_targets() -> None:
    feature = np.arange(4 * 2 * 2, dtype=np.float32).reshape(4, 2, 2)

    samples = build_matcha_joint_index_training_set_from_maps(
        feature,
        feature,
        _toy_robust_supervision(),
        hard_negatives_per_match=2,
    )

    base = samples.coarse_fine_samples
    assert base.query_offset_soft_labels is not None
    assert base.query_offset_soft_labels.shape[1] == 65
    assert np.allclose(base.query_offset_soft_labels[:2, 36], 0.8)
    assert base.sample_confidence_targets is not None
    assert np.allclose(base.sample_confidence_targets[:3], [1.0, 0.35, 0.0])
    assert base.sample_uncertainty_px is not None
    assert base.sample_uncertainty_px[1] == 4.0
    assert samples.sample_no_match_labels is not None
    assert samples.sample_no_match_labels[:3].tolist() == [0, 0, 1]
    assert samples.sample_ignore_mask is not None
    assert samples.sample_ignore_mask.shape[0] == base.sample_count
    assert samples.sample_confidence_ignore_mask is not None
    assert samples.sample_confidence_ignore_mask.tolist() == [False, True, True, False, False]


def test_build_matcha_joint_index_training_set_omits_rgb_when_keypoint_labels_absent() -> None:
    feature = np.arange(4 * 2 * 2, dtype=np.float32).reshape(4, 2, 2)
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)

    samples = build_matcha_joint_index_training_set_from_maps(
        feature,
        feature,
        _toy_supervision(),
        query_rgb=rgb,
        render_rgb=rgb,
        query_keypoint_label_map=None,
        render_keypoint_label_map=None,
        hard_negatives_per_match=2,
    )

    assert samples.query_rgb_images is None
    assert samples.render_rgb_images is None


def test_matcha_joint_cache_extractor_can_build_radio_dual_feature_map() -> None:
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)

    feature = _extract_matcha_joint_feature_from_rgb(
        rgb,
        _FakeRadioExtractor(),
        feature_mode="radio_dual",
        fine_intermediate_index=-6,
        coarse_source="final",
        coarse_intermediate_index=-1,
    )

    assert feature.shape == (8, 2, 2)
    assert np.allclose(feature[:4], 1.0)
    assert np.allclose(feature[4:], 2.0)


def test_build_matcha_joint_training_set_adds_repeatability_and_match_masks() -> None:
    feature_map = np.eye(8, dtype=np.float32).T.reshape(8, 2, 4)
    supervision = _toy_supervision()

    samples = build_matcha_joint_training_set_from_maps(
        feature_map,
        feature_map.copy(),
        supervision,
        hard_negatives_per_match=1,
        roundtrip_heatmap_threshold_px=2.0,
    )

    assert samples.sample_no_match_labels is not None
    assert samples.sample_no_match_labels.shape == (samples.coarse_fine_samples.sample_count,)
    assert np.count_nonzero(samples.sample_no_match_labels) == supervision.count
    assert samples.sample_ignore_mask is not None
    assert np.count_nonzero(samples.sample_ignore_mask) == 0
    assert np.all(samples.coarse_fine_samples.query_offset_labels[samples.sample_no_match_labels == 1] == 64)
    assert np.all(samples.coarse_fine_samples.render_offset_labels[samples.sample_no_match_labels == 1] == 64)
    assert samples.query_repeatability_targets is not None
    assert samples.render_repeatability_targets is not None
    assert np.max(samples.query_repeatability_targets) <= 1.0
    assert np.max(samples.render_repeatability_targets) <= 1.0


def test_coarse_fine_confidence_bce_skips_confidence_ignored_rows() -> None:
    samples = MatchaCoarseFineTrainingSet(
        query_features=np.asarray([[10.0], [-10.0], [-10.0]], dtype=np.float32),
        render_features=np.zeros((3, 1), dtype=np.float32),
        query_offset_labels=np.zeros((3,), dtype=np.int64),
        render_offset_labels=np.zeros((3,), dtype=np.int64),
        negative_render_features=np.zeros((3, 1, 1), dtype=np.float32),
        roundtrip_errors_px=np.zeros((3,), dtype=np.float32),
        sample_confidence_targets=np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
    )
    config = MatchaJointTrainingConfig(
        output_dim=1,
        residual_hidden_dim=4,
        batch_size=3,
        dual_softmax_weight=0.0,
        offset_loss_weight=0.0,
        pair_fine_loss_weight=0.0,
        query_pair_fine_loss_weight=0.0,
        pair_confidence_loss_weight=1.0,
        hard_negative_weight=0.0,
        group_size=1,
    )

    loss, _state = joint_training._coarse_fine_loss(
        _ConfidenceOnlyModel(),
        samples,
        np.arange(3, dtype=np.int64),
        config,
        torch.device("cpu"),
        confidence_ignore_mask=np.asarray([False, True, False]),
    )

    expected_pos = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor([10.0, -10.0]),
        torch.tensor([1.0, 0.0]),
    )
    expected_neg = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor([10.0, -10.0, -10.0]),
        torch.zeros(3),
    )
    assert torch.allclose(loss, expected_pos + expected_neg)


def test_no_match_confidence_loss_uses_sample_targets_for_pose_usable_rows() -> None:
    samples = MatchaJointTrainingSet(
        coarse_fine_samples=MatchaCoarseFineTrainingSet(
            query_features=np.asarray([[0.0], [10.0], [-10.0]], dtype=np.float32),
            render_features=np.zeros((3, 1), dtype=np.float32),
            query_offset_labels=np.asarray([0, 64, 64], dtype=np.int64),
            render_offset_labels=np.asarray([0, 64, 64], dtype=np.int64),
            negative_render_features=np.zeros((3, 0, 1), dtype=np.float32),
            roundtrip_errors_px=np.zeros((3,), dtype=np.float32),
            sample_confidence_targets=np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
        ),
        sample_no_match_labels=np.asarray([0, 1, 1], dtype=np.int64),
        sample_confidence_ignore_mask=np.asarray([False, False, False], dtype=bool),
    )

    loss = joint_training._no_match_confidence_loss(
        _ConfidenceOnlyModel(),
        samples,
        np.arange(3, dtype=np.int64),
        device=torch.device("cpu"),
    )

    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor([10.0, -10.0]),
        torch.tensor([1.0, 0.0]),
    )
    assert loss is not None
    assert torch.allclose(loss, expected)


def test_index_only_joint_cache_round_trip_omits_row_descriptor_tensors(tmp_path) -> None:
    feature_map = np.eye(8, dtype=np.float32).T.reshape(8, 2, 4)
    supervision = _toy_supervision()
    cache_path = tmp_path / "index_only_joint.npz"

    samples = build_matcha_joint_index_training_set_from_maps(
        feature_map,
        feature_map.copy(),
        supervision,
        hard_negatives_per_match=2,
        roundtrip_heatmap_threshold_px=2.0,
    )
    save_matcha_joint_training_set_npz(samples, cache_path)

    with np.load(cache_path, allow_pickle=True) as data:
        assert "query_features" not in data.files
        assert "render_features" not in data.files
        assert "negative_render_features" not in data.files
        assert "negative_render_indices" in data.files

    loaded, metadata = load_matcha_joint_training_set_npz(cache_path)
    assert metadata["format"] == "vfm_matcha_joint_index_training_set_v2"
    assert loaded.coarse_fine_samples.sample_count == 8
    assert loaded.coarse_fine_samples.input_dim == 8
    assert loaded.sample_no_match_labels is not None
    assert loaded.sample_no_match_labels.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert loaded.coarse_fine_samples.query_features[[0, 4]].shape == (2, 8)
    assert loaded.coarse_fine_samples.negative_render_features[[0, 4]].shape == (2, 2, 8)
    assert np.allclose(loaded.coarse_fine_samples.query_features[[0]], feature_map.reshape(8, -1).T[[0]])


def test_train_matcha_joint_model_accepts_index_only_dynamic_gather() -> None:
    feature_map = np.eye(8, dtype=np.float32).T.reshape(8, 2, 4)
    samples = build_matcha_joint_index_training_set_from_maps(
        feature_map,
        feature_map.copy(),
        _toy_supervision(),
        hard_negatives_per_match=2,
        roundtrip_heatmap_threshold_px=2.0,
    )

    run = train_matcha_joint_model(
        samples,
        MatchaJointTrainingConfig(
            output_dim=8,
            residual_hidden_dim=16,
            steps=1,
            batch_size=4,
            dense_heatmap_loss_weight=0.1,
            hard_false_match_weight=0.1,
            device="cpu",
            seed=23,
        ),
    )

    assert run.summary["sample_count"] == 8
    assert run.summary["input_dim"] == 8
    assert "map_descriptor_top1_acc" in run.summary


def test_matcha_style_joint_model_forwards_all_matcha_heads() -> None:
    model = MatchaStyleJointModel(input_dim=8, output_dim=8, residual_hidden_dim=16, group_size=4)
    feature_maps = torch.eye(8, dtype=torch.float32).T.reshape(1, 8, 2, 4)
    images = torch.rand(1, 3, 16, 32)

    descriptors, heatmap_logits, offset_logits = model.forward_feature_map(feature_maps)
    keypoint_logits = model.forward_rgb_keypoints(images)

    assert descriptors.shape == (1, 8, 2, 4)
    assert heatmap_logits.shape == (1, 1, 2, 4)
    assert offset_logits.shape == (1, 65, 2, 4)
    assert keypoint_logits.shape == (1, 65, 2, 4)


def test_radio_dual_attention_fusion_model_splits_fine_and_coarse_maps() -> None:
    model = RadioDualAttentionFusionJointModel(
        fine_input_dim=4,
        coarse_input_dim=4,
        output_dim=4,
        residual_hidden_dim=16,
        attention_hidden_dim=8,
        attention_depth=1,
        attention_heads=2,
        attention_patch_size=2,
        group_size=4,
    )
    fine = torch.eye(4, dtype=torch.float32).T.reshape(1, 4, 2, 2)
    coarse = torch.flip(fine, dims=[1])
    feature_maps = torch.cat([fine, coarse], dim=1)
    images = torch.rand(1, 3, 16, 16)

    descriptors, heatmap_logits, offset_logits = model.forward_feature_map(feature_maps)
    keypoint_logits = model.forward_rgb_keypoints(images)
    query_z = descriptors.permute(0, 2, 3, 1).reshape(-1, 4)
    pair_logits = model.pair_fine_logits(query_z, query_z)
    confidence_logits = model.pair_confidence_logits(query_z, query_z)

    assert descriptors.shape == (1, 4, 2, 2)
    assert heatmap_logits.shape == (1, 1, 2, 2)
    assert offset_logits.shape == (1, 65, 2, 2)
    assert keypoint_logits.shape == (1, 65, 2, 2)
    assert pair_logits.shape == (4, 64)
    assert confidence_logits.shape == (4,)
    norms = torch.linalg.norm(descriptors.reshape(4, -1), dim=0)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_radio_dual_attention_fusion_model_uses_downsampled_attention_tokens() -> None:
    model = RadioDualAttentionFusionJointModel(
        fine_input_dim=4,
        coarse_input_dim=4,
        output_dim=4,
        residual_hidden_dim=16,
        attention_hidden_dim=8,
        attention_depth=1,
        attention_heads=2,
        attention_patch_size=2,
        group_size=4,
    )
    feature_maps = torch.randn(1, 8, 4, 6)

    descriptors, heatmap_logits, offset_logits = model.forward_feature_map(feature_maps)

    assert model.attention_patch_size == 2
    assert descriptors.shape == (1, 4, 4, 6)
    assert heatmap_logits.shape == (1, 1, 4, 6)
    assert offset_logits.shape == (1, 65, 4, 6)


def test_radio_dual_attention_fusion_model_supports_pixel_shuffle_context() -> None:
    model = RadioDualAttentionFusionJointModel(
        fine_input_dim=4,
        coarse_input_dim=4,
        output_dim=4,
        residual_hidden_dim=16,
        attention_hidden_dim=8,
        attention_depth=1,
        attention_heads=2,
        attention_patch_size=2,
        attention_upsample_mode="pixel_shuffle",
        group_size=4,
    )
    feature_maps = torch.randn(1, 8, 4, 6)

    descriptors, heatmap_logits, offset_logits = model.forward_feature_map(feature_maps)

    assert model.attention_upsample_mode == "pixel_shuffle"
    assert descriptors.shape == (1, 4, 4, 6)
    assert heatmap_logits.shape == (1, 1, 4, 6)
    assert offset_logits.shape == (1, 65, 4, 6)


def test_radio_dual_attention_fusion_model_exposes_matcha_original_fine_and_coarse_outputs() -> None:
    model = RadioDualAttentionFusionJointModel(
        fine_input_dim=4,
        coarse_input_dim=4,
        output_dim=4,
        residual_hidden_dim=16,
        attention_hidden_dim=16,
        attention_depth=1,
        attention_heads=2,
        attention_patch_size=2,
        attention_upsample_mode="pixel_shuffle",
        attention_fusion_mode="matcha_original",
        group_size=4,
    )
    fine = torch.eye(4, dtype=torch.float32).T.reshape(1, 4, 2, 2).repeat(1, 1, 2, 3)
    coarse = torch.flip(fine, dims=[1])
    feature_maps = torch.cat([fine, coarse], dim=1)
    images = torch.rand(1, 3, 32, 48)

    coarse_descriptors, fine_descriptors, heatmap_logits = model.forward_fuse_feature(feature_maps)
    descriptors, heatmap_logits_from_forward, offset_logits = model.forward_feature_map(feature_maps)
    keypoint_logits = model.forward_rgb_keypoints(images)
    rows = descriptors.permute(0, 2, 3, 1).reshape(-1, 4)
    pair_logits = model.pair_fine_logits(rows, rows)
    confidence_logits = model.pair_confidence_logits(rows, rows)

    assert coarse_descriptors.shape == (1, 4, 4, 6)
    assert fine_descriptors.shape == (1, 4, 4, 6)
    assert descriptors.shape == fine_descriptors.shape
    assert torch.allclose(descriptors, fine_descriptors)
    assert heatmap_logits.shape == (1, 1, 4, 6)
    assert torch.allclose(heatmap_logits, heatmap_logits_from_forward)
    assert offset_logits.shape == (1, 65, 4, 6)
    assert keypoint_logits.shape == (1, 65, 4, 6)
    assert pair_logits.shape == (24, 64)
    assert confidence_logits.shape == (24,)
    assert torch.allclose(torch.linalg.norm(coarse_descriptors, dim=1), torch.ones(1, 4, 6), atol=1e-5)
    assert torch.allclose(torch.linalg.norm(fine_descriptors, dim=1), torch.ones(1, 4, 6), atol=1e-5)


def test_train_matcha_joint_model_optimizes_descriptor_heatmap_and_rgb_detector() -> None:
    feature_map = np.eye(8, dtype=np.float32).T.reshape(1, 8, 2, 4)
    heatmap_target = np.zeros((1, 2, 4), dtype=np.float32)
    heatmap_target.reshape(-1)[:4] = 1.0
    labels = np.arange(8, dtype=np.int64).reshape(1, 2, 4)
    rgb = np.zeros((1, 3, 16, 32), dtype=np.float32)
    for cell, label in enumerate(labels.reshape(-1).tolist()):
        row, col = divmod(cell, 4)
        y = row * 8 + int(label) // 8
        x = col * 8 + int(label) % 8
        rgb[0, :, y, x] = 1.0

    run = train_matcha_joint_model(
        MatchaJointTrainingSet(
            coarse_fine_samples=_toy_samples(),
            query_feature_maps=feature_map,
            render_feature_maps=feature_map.copy(),
            query_heatmap_targets=heatmap_target,
            render_heatmap_targets=heatmap_target.copy(),
            query_cell_indices=np.asarray([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64),
            render_cell_indices=np.asarray([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64),
            query_rgb_images=rgb,
            render_rgb_images=rgb.copy(),
            query_rgb_keypoint_labels=labels,
            render_rgb_keypoint_labels=labels.copy(),
        ),
        MatchaJointTrainingConfig(
            output_dim=8,
            residual_hidden_dim=16,
            steps=100,
            batch_size=8,
            lr=5e-3,
            dual_softmax_weight=1.0,
            offset_loss_weight=0.5,
            pair_fine_loss_weight=0.25,
            pair_confidence_loss_weight=0.2,
            dense_heatmap_loss_weight=0.5,
            rgb_keypoint_loss_weight=0.5,
            local_fine_transformer_loss_weight=0.25,
            rgb_keypoint_position_loss_weight=0.25,
            hard_negative_weight=0.0,
            group_size=4,
            device="cpu",
            seed=3,
        ),
    )

    assert run.summary["final_loss"] < run.summary["initial_loss"]
    assert run.summary["train_top1_acc"] >= 0.99
    assert run.summary["map_descriptor_top1_acc"] >= 0.99
    assert run.summary["query_offset_acc"] >= 0.75
    assert run.summary["render_offset_acc"] >= 0.75
    assert run.summary["query_heatmap_mae"] < 0.35
    assert run.summary["render_heatmap_mae"] < 0.35
    assert run.summary["local_fine_transformer_acc"] >= 0.75
    assert run.summary["query_rgb_position_acc"] >= 0.99
    assert run.summary["render_rgb_position_acc"] >= 0.99
    assert run.summary["query_rgb_keypoint_positive_acc"] >= 0.99
    assert run.summary["render_rgb_keypoint_positive_acc"] >= 0.99


def test_train_matcha_joint_model_can_supervise_query_pair_fine_head() -> None:
    features = np.eye(8, dtype=np.float32)
    samples = MatchaJointTrainingSet(
        coarse_fine_samples=MatchaCoarseFineTrainingSet(
            query_features=features,
            render_features=features,
            query_offset_labels=np.arange(8, dtype=np.int64),
            render_offset_labels=np.full((8,), 36, dtype=np.int64),
            negative_render_features=np.roll(features, shift=1, axis=0)[:, None, :],
            roundtrip_errors_px=np.zeros((8,), dtype=np.float32),
            metadata={"toy": True},
        )
    )

    run = train_matcha_joint_model(
        samples,
        MatchaJointTrainingConfig(
            output_dim=8,
            residual_hidden_dim=16,
            steps=120,
            batch_size=8,
            lr=5e-3,
            dual_softmax_weight=0.25,
            offset_loss_weight=0.0,
            pair_fine_loss_weight=0.0,
            query_pair_fine_loss_weight=1.0,
            pair_confidence_loss_weight=0.0,
            dense_heatmap_loss_weight=0.0,
            rgb_keypoint_loss_weight=0.0,
            hard_negative_weight=0.0,
            group_size=4,
            device="cpu",
            seed=33,
        ),
    )

    assert run.summary["query_pair_fine_acc"] >= 0.75


def test_train_matcha_joint_model_records_best_validation_checkpoint() -> None:
    train_samples = MatchaJointTrainingSet(coarse_fine_samples=_toy_samples())
    validation_samples = MatchaJointTrainingSet(coarse_fine_samples=_toy_samples())

    run = train_matcha_joint_model(
        train_samples,
        MatchaJointTrainingConfig(
            output_dim=8,
            residual_hidden_dim=16,
            steps=2,
            batch_size=8,
            lr=5e-3,
            dual_softmax_weight=1.0,
            offset_loss_weight=0.0,
            pair_fine_loss_weight=0.0,
            pair_confidence_loss_weight=0.0,
            dense_heatmap_loss_weight=0.0,
            rgb_keypoint_loss_weight=0.0,
            hard_negative_weight=0.0,
            group_size=4,
            device="cpu",
            seed=51,
        ),
        validation_samples=validation_samples,
        validation_interval=1,
    )

    assert "best_validation_loss" in run.summary
    assert "best_validation_step" in run.summary
    assert run.summary["validation_eval_count"] >= 2
    assert run.summary["best_validation_step"] in {0, 1, 2}


def test_train_matcha_joint_model_from_manifest_loads_shards_lazily(tmp_path, monkeypatch) -> None:
    shards = [MatchaJointTrainingSet(coarse_fine_samples=_toy_samples()) for _ in range(3)]
    manifest = tmp_path / "joint_manifest.json"
    save_matcha_joint_training_set_manifest(shards, manifest)
    original_loader = joint_training.load_matcha_joint_training_set_npz
    loaded_paths: list[str] = []

    def counting_loader(path):
        loaded_paths.append(str(path))
        return original_loader(path)

    monkeypatch.setattr(joint_training, "load_matcha_joint_training_set_npz", counting_loader)

    run = train_matcha_joint_model_from_manifest(
        manifest,
        MatchaJointTrainingConfig(
            output_dim=8,
            residual_hidden_dim=16,
            steps=1,
            batch_size=8,
            lr=5e-3,
            dual_softmax_weight=1.0,
            offset_loss_weight=0.0,
            pair_fine_loss_weight=0.0,
            pair_confidence_loss_weight=0.0,
            dense_heatmap_loss_weight=0.0,
            rgb_keypoint_loss_weight=0.0,
            hard_negative_weight=0.0,
            group_size=4,
            device="cpu",
            seed=52,
        ),
        shard_cache_size=1,
        steps_per_shard=1,
    )

    assert run.summary["manifest_shard_count"] == 3
    assert len(set(loaded_paths)) < 3


def test_train_matcha_joint_model_uses_no_match_ignore_and_repeatability_targets() -> None:
    feature_map = np.eye(8, dtype=np.float32).T.reshape(1, 8, 2, 4)
    heatmap_target = np.ones((1, 2, 4), dtype=np.float32)
    rgb = np.zeros((1, 3, 16, 32), dtype=np.float32)
    keypoint_labels = np.full((1, 2, 4), 64, dtype=np.int64)
    samples = MatchaJointTrainingSet(
        coarse_fine_samples=_toy_samples(),
        query_feature_maps=feature_map,
        render_feature_maps=feature_map.copy(),
        query_heatmap_targets=heatmap_target,
        render_heatmap_targets=heatmap_target.copy(),
        query_rgb_images=rgb,
        render_rgb_images=rgb.copy(),
        query_rgb_keypoint_labels=keypoint_labels,
        render_rgb_keypoint_labels=keypoint_labels.copy(),
        query_cell_indices=np.arange(8, dtype=np.int64),
        render_cell_indices=np.arange(8, dtype=np.int64),
        sample_no_match_labels=np.asarray([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.int64),
        sample_ignore_mask=np.asarray([False, False, True, False, False, False, False, False]),
        query_repeatability_targets=np.ones((1, 2, 4), dtype=np.float32),
        render_repeatability_targets=np.zeros((1, 2, 4), dtype=np.float32),
    )

    run = train_matcha_joint_model(
        samples,
        MatchaJointTrainingConfig(
            output_dim=8,
            residual_hidden_dim=16,
            steps=2,
            batch_size=8,
            lr=1e-3,
            dual_softmax_weight=1.0,
            offset_loss_weight=0.0,
            pair_fine_loss_weight=0.0,
            pair_confidence_loss_weight=0.1,
            dense_heatmap_loss_weight=0.0,
            rgb_keypoint_loss_weight=0.0,
            repeatability_loss_weight=0.1,
            hard_negative_weight=0.0,
            group_size=4,
            device="cpu",
            seed=62,
        ),
    )

    assert "no_match_confidence_loss" in run.summary
    assert "query_repeatability_loss" in run.summary
    assert run.summary["positive_match_count"] >= 1


def test_train_matcha_joint_model_reports_hard_false_match_mining_loss() -> None:
    feature_map = np.eye(8, dtype=np.float32).T.reshape(1, 8, 2, 4)
    samples = MatchaJointTrainingSet(
        coarse_fine_samples=_toy_samples(),
        query_feature_maps=feature_map,
        render_feature_maps=feature_map.copy(),
        query_heatmap_targets=np.ones((1, 2, 4), dtype=np.float32),
        render_heatmap_targets=np.ones((1, 2, 4), dtype=np.float32),
        query_cell_indices=np.arange(8, dtype=np.int64),
        render_cell_indices=np.arange(8, dtype=np.int64),
    )

    run = train_matcha_joint_model(
        samples,
        MatchaJointTrainingConfig(
            output_dim=8,
            residual_hidden_dim=16,
            steps=2,
            batch_size=8,
            lr=1e-3,
            dual_softmax_weight=1.0,
            offset_loss_weight=0.0,
            pair_fine_loss_weight=0.0,
            pair_confidence_loss_weight=0.0,
            dense_heatmap_loss_weight=0.0,
            rgb_keypoint_loss_weight=0.0,
            hard_negative_weight=0.0,
            hard_false_match_weight=0.2,
            group_size=4,
            device="cpu",
            seed=63,
        ),
    )

    assert "hard_false_match_loss" in run.summary
    assert run.summary["hard_false_match_count"] > 0


def test_train_matcha_joint_model_can_warm_start_from_existing_joint_model() -> None:
    samples = MatchaJointTrainingSet(coarse_fine_samples=_toy_samples())
    config = MatchaJointTrainingConfig(
        output_dim=8,
        residual_hidden_dim=16,
        steps=1,
        batch_size=8,
        lr=1e-3,
        dual_softmax_weight=1.0,
        offset_loss_weight=0.0,
        pair_fine_loss_weight=0.0,
        pair_confidence_loss_weight=0.0,
        dense_heatmap_loss_weight=0.0,
        rgb_keypoint_loss_weight=0.0,
        hard_negative_weight=0.0,
        group_size=4,
        device="cpu",
        seed=64,
    )
    source = train_matcha_joint_model(samples, config)

    run = train_matcha_joint_model(samples, config, warm_start_model=source.model)

    assert run.summary["warm_start_loaded"] is True
    assert run.summary["warm_start_missing_keys"] == []
    assert run.summary["warm_start_unexpected_keys"] == []


def test_local_patch_correlation_loss_prefers_window_center_match() -> None:
    query_desc = torch.zeros(1, 4, 3, 3)
    render_desc = torch.zeros(1, 4, 3, 3)
    query_desc[0, :, 1, 1] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    render_desc[0, :, 1, 1] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    render_desc[0, :, 1, 2] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    query_z = query_desc.permute(0, 2, 3, 1).reshape(1, 9, 4)[:, 4]
    pairs = torch.asarray([0], dtype=torch.long)
    target_indices = torch.asarray([4], dtype=torch.long)

    loss, acc = joint_training._local_patch_correlation_loss(
        source_descriptors=query_z,
        target_descriptor_map=render_desc,
        pair_indices=pairs,
        target_cell_indices=target_indices,
        window_size=3,
    )

    assert loss is not None
    assert float(loss.item()) < 1.5
    assert acc == 1.0


def test_local_patch_correlation_loss_ignores_boundary_windows() -> None:
    descriptors = torch.randn(1, 4, 3, 3)
    source = descriptors.permute(0, 2, 3, 1).reshape(1, 9, 4)[:, 0]

    loss, acc = joint_training._local_patch_correlation_loss(
        source_descriptors=source,
        target_descriptor_map=descriptors,
        pair_indices=torch.asarray([0], dtype=torch.long),
        target_cell_indices=torch.asarray([0], dtype=torch.long),
        window_size=3,
    )

    assert loss is None
    assert acc == 0.0


def test_matcha_rgb_keypoint_position_loss_uses_valid_correspondence_mask() -> None:
    query_logits = torch.full((1, 65, 1, 2), -10.0)
    render_logits = torch.full((1, 65, 1, 2), -10.0)
    query_logits[0, 0, 0, 0] = 10.0
    query_logits[0, 1, 0, 1] = 10.0
    render_logits[0, 9, 0, 0] = 10.0
    render_logits[0, 9, 0, 1] = 10.0
    pts1 = torch.asarray([[0.0, 0.0]], dtype=torch.float32)
    pts2 = torch.asarray([[1.0, 1.0]], dtype=torch.float32)

    loss, acc, metrics = matcha_keypoint_position_loss(query_logits, render_logits, pts1, pts2)

    assert metrics["valid_count"] == 1
    assert metrics["source_candidate_count"] == 2
    assert acc == 1.0
    assert float(loss.item()) < 1e-3


def test_matcha_rgb_keypoint_position_loss_ignores_dustbin_predictions_and_labels() -> None:
    query_logits = torch.full((1, 65, 1, 3), -10.0)
    render_logits = torch.full((1, 65, 1, 3), -10.0)
    query_logits[0, 64, 0, 0] = 20.0
    query_logits[0, 5, 0, 1] = 20.0
    query_logits[0, 0, 0, 2] = 20.0
    render_logits[0, 64, 0, 0] = 20.0
    render_logits[0, 64, 0, 1] = 20.0
    render_logits[0, 5, 0, 2] = 20.0
    pts1 = torch.asarray([[0.0, 0.0], [13.0, 0.0], [16.0, 0.0]], dtype=torch.float32)
    pts2 = torch.asarray([[64.0, 0.0], [64.0, 0.0], [21.0, 0.0]], dtype=torch.float32)

    loss, acc, metrics = matcha_keypoint_position_loss(query_logits, render_logits, pts1, pts2)

    assert metrics["source_candidate_count"] == 2
    assert metrics["valid_count"] == 1
    assert acc == 1.0
    assert float(loss.item()) < 1e-3


def test_matcha_rgb_keypoint_position_loss_prefers_roundtrip_offsets_over_cell_labels() -> None:
    query_logits = torch.full((1, 65, 1, 1), -10.0)
    good_render_logits = torch.full((1, 65, 1, 1), -10.0)
    bad_render_logits = torch.full((1, 65, 1, 1), -10.0)
    query_logits[0, 0, 0, 0] = 10.0
    good_render_logits[0, 9, 0, 0] = 10.0
    bad_render_logits[0, 0, 0, 0] = 10.0
    pts1 = torch.asarray([[0.0, 0.0]], dtype=torch.float32)
    pts2 = torch.asarray([[1.0, 1.0]], dtype=torch.float32)

    good_loss, good_acc, _metrics = matcha_keypoint_position_loss(query_logits, good_render_logits, pts1, pts2)
    bad_loss, bad_acc, _bad_metrics = matcha_keypoint_position_loss(query_logits, bad_render_logits, pts1, pts2)

    assert good_acc == 1.0
    assert bad_acc == 0.0
    assert float(good_loss.item()) + 10.0 < float(bad_loss.item())


def test_train_matcha_joint_model_can_use_radio_dual_attention_architecture(tmp_path) -> None:
    base_features = np.eye(4, dtype=np.float32)
    fine_map = base_features.T.reshape(1, 4, 2, 2)
    coarse_map = np.flip(fine_map, axis=1).copy()
    feature_map = np.concatenate(
        [
            fine_map,
            coarse_map,
        ],
        axis=1,
    )
    heatmap_target = np.ones((1, 2, 2), dtype=np.float32)
    labels = np.arange(4, dtype=np.int64).reshape(1, 2, 2)
    rgb = np.zeros((1, 3, 16, 16), dtype=np.float32)
    samples = MatchaJointTrainingSet(
        coarse_fine_samples=MatchaCoarseFineTrainingSet(
            query_features=np.concatenate([base_features, np.flip(base_features, axis=1).copy()], axis=1),
            render_features=np.concatenate([base_features, np.flip(base_features, axis=1).copy()], axis=1),
            query_offset_labels=np.asarray([0, 1, 2, 3], dtype=np.int64),
            render_offset_labels=np.asarray([0, 1, 2, 3], dtype=np.int64),
            negative_render_features=np.roll(
                np.concatenate([base_features, np.flip(base_features, axis=1).copy()], axis=1),
                shift=1,
                axis=0,
            )[:, None, :],
            roundtrip_errors_px=np.zeros((4,), dtype=np.float32),
            metadata={"toy": True},
        ),
        query_feature_maps=feature_map,
        render_feature_maps=feature_map.copy(),
        query_heatmap_targets=heatmap_target,
        render_heatmap_targets=heatmap_target.copy(),
        query_cell_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
        render_cell_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
        query_rgb_images=rgb,
        render_rgb_images=rgb.copy(),
        query_rgb_keypoint_labels=labels,
        render_rgb_keypoint_labels=labels.copy(),
    )

    run = train_matcha_joint_model(
        samples,
        MatchaJointTrainingConfig(
            model_type="radio_dual_attention",
            fine_input_dim=4,
            coarse_input_dim=4,
            output_dim=4,
            residual_hidden_dim=16,
            attention_hidden_dim=8,
            attention_depth=1,
            attention_heads=2,
            attention_upsample_mode="pixel_shuffle",
            attention_fusion_mode="matcha_original",
            steps=2,
            batch_size=4,
            lr=1e-3,
            group_size=4,
            device="cpu",
            seed=11,
        ),
    )
    joint_path = tmp_path / "radio_dual_joint.pt"
    save_matcha_joint_model(run, joint_path)
    loaded = load_matcha_joint_model(joint_path, device="cpu")
    selected, offsets, heatmap = project_feature_map_with_matcha_joint_model(
        loaded.model,
        feature_map[0],
        device="cpu",
    )

    assert loaded.summary["model_type"] == "radio_dual_attention"
    assert loaded.model.attention_fusion_mode == "matcha_original"
    assert selected.shape == (4, 2, 2)
    assert offsets.shape == (65, 2, 2)
    assert heatmap.shape == (2, 2)


def test_matcha_joint_training_set_npz_round_trip(tmp_path) -> None:
    feature_map = np.eye(8, dtype=np.float32).T.reshape(1, 8, 2, 4)
    heatmap_target = np.zeros((1, 2, 4), dtype=np.float32)
    rgb = np.zeros((1, 3, 16, 32), dtype=np.float32)
    labels = np.full((1, 2, 4), 64, dtype=np.int64)
    labels[0, 0, 0] = 36
    original = MatchaJointTrainingSet(
        coarse_fine_samples=_toy_samples(),
        query_feature_maps=feature_map,
        render_feature_maps=feature_map.copy(),
        query_heatmap_targets=heatmap_target,
        render_heatmap_targets=heatmap_target.copy(),
        query_rgb_images=rgb,
        render_rgb_images=rgb.copy(),
        query_rgb_keypoint_labels=labels,
        render_rgb_keypoint_labels=labels.copy(),
    )

    path = tmp_path / "joint_cache.npz"
    save_matcha_joint_training_set_npz(original, path)
    loaded, metadata = load_matcha_joint_training_set_npz(path)

    assert metadata["format"] == "vfm_matcha_joint_training_set_v1"
    assert loaded.coarse_fine_samples.sample_count == original.coarse_fine_samples.sample_count
    assert loaded.query_feature_maps is not None
    assert loaded.query_feature_maps.shape == (1, 8, 2, 4)
    assert loaded.query_rgb_keypoint_labels is not None
    assert loaded.query_rgb_keypoint_labels[0, 0, 0] == 36


def test_matcha_joint_training_set_npz_round_trip_perturbation_fields(tmp_path) -> None:
    feature_map = np.eye(8, dtype=np.float32).T.reshape(1, 8, 2, 4)
    heatmap_target = np.zeros((1, 2, 4), dtype=np.float32)
    original = MatchaJointTrainingSet(
        coarse_fine_samples=_toy_samples(),
        query_feature_maps=feature_map,
        render_feature_maps=feature_map.copy(),
        query_heatmap_targets=heatmap_target,
        render_heatmap_targets=heatmap_target.copy(),
        query_cell_indices=np.arange(8, dtype=np.int64),
        render_cell_indices=np.arange(8, dtype=np.int64),
        pair_type_ids=np.asarray([2], dtype=np.int64),
        pair_type_names=np.asarray(["C_trans050"], dtype=object),
        pair_query_ids=np.asarray(["seq8/frame00001.png"], dtype=object),
        pair_split_names=np.asarray(["train"], dtype=object),
        pair_candidate_ids=np.asarray([""], dtype=object),
        pair_translation_errors_m=np.asarray([0.5], dtype=np.float32),
        pair_rotation_errors_deg=np.asarray([3.0], dtype=np.float32),
        sample_no_match_labels=np.asarray([0, 1, 0, 1, 0, 0, 1, 0], dtype=np.int64),
        sample_ignore_mask=np.asarray([False, False, True, False, False, False, True, False]),
        sample_confidence_ignore_mask=np.asarray([False, True, True, False, False, False, True, False]),
        query_repeatability_targets=np.full((1, 2, 4), 0.25, dtype=np.float32),
        render_repeatability_targets=np.full((1, 2, 4), 0.5, dtype=np.float32),
    )

    path = tmp_path / "joint_cache_perturb.npz"
    save_matcha_joint_training_set_npz(original, path)
    loaded, _metadata = load_matcha_joint_training_set_npz(path)

    assert loaded.pair_type_ids is not None
    assert loaded.pair_type_ids.tolist() == [2]
    assert loaded.pair_type_names is not None
    assert loaded.pair_type_names.tolist() == ["C_trans050"]
    assert loaded.pair_query_ids is not None
    assert loaded.pair_query_ids.tolist() == ["seq8/frame00001.png"]
    assert loaded.pair_split_names is not None
    assert loaded.pair_split_names.tolist() == ["train"]
    assert loaded.pair_candidate_ids is not None
    assert loaded.pair_candidate_ids.tolist() == [""]
    assert loaded.pair_translation_errors_m is not None
    assert np.allclose(loaded.pair_translation_errors_m, [0.5])
    assert loaded.sample_no_match_labels is not None
    assert loaded.sample_no_match_labels.tolist() == [0, 1, 0, 1, 0, 0, 1, 0]
    assert loaded.sample_ignore_mask is not None
    assert loaded.sample_ignore_mask.tolist() == [False, False, True, False, False, False, True, False]
    assert loaded.sample_confidence_ignore_mask is not None
    assert loaded.sample_confidence_ignore_mask.tolist() == [False, True, True, False, False, False, True, False]
    assert loaded.query_repeatability_targets is not None
    assert np.allclose(loaded.query_repeatability_targets, 0.25)
    assert loaded.render_repeatability_targets is not None
    assert np.allclose(loaded.render_repeatability_targets, 0.5)


def test_matcha_joint_training_set_manifest_round_trip_merges_shards(tmp_path) -> None:
    feature_map = np.eye(8, dtype=np.float32).T.reshape(8, 2, 4)
    one = MatchaJointTrainingSet(
        coarse_fine_samples=_toy_samples(),
        query_feature_maps=feature_map[None],
        render_feature_maps=feature_map[None],
        query_heatmap_targets=np.ones((1, 2, 4), dtype=np.float32),
        render_heatmap_targets=np.ones((1, 2, 4), dtype=np.float32),
        query_cell_indices=np.arange(8, dtype=np.int64),
        render_cell_indices=np.arange(8, dtype=np.int64),
        pair_type_ids=np.asarray([0], dtype=np.int64),
        pair_type_names=np.asarray(["A_gt"], dtype=object),
        pair_query_ids=np.asarray(["seq8/frame00001.png"], dtype=object),
        pair_split_names=np.asarray(["train"], dtype=object),
        pair_candidate_ids=np.asarray([""], dtype=object),
    )
    manifest = tmp_path / "joint_manifest.json"

    metadata = save_matcha_joint_training_set_manifest([one, one], manifest)
    loaded, loaded_metadata = load_matcha_joint_training_set_manifest(manifest)

    assert metadata["format"] == "vfm_matcha_joint_training_manifest_v1"
    assert loaded_metadata["format"] == "vfm_matcha_joint_training_manifest_v1"
    assert loaded_metadata["shard_count"] == 2
    assert loaded.coarse_fine_samples.sample_count == 16
    assert loaded.query_feature_maps is not None
    assert loaded.query_feature_maps.shape == (2, 8, 2, 4)
    assert loaded.sample_pair_indices is not None
    assert loaded.sample_pair_indices[:8].tolist() == [0] * 8
    assert loaded.sample_pair_indices[8:].tolist() == [1] * 8
    assert loaded.pair_query_ids is not None
    assert loaded.pair_query_ids.tolist() == ["seq8/frame00001.png", "seq8/frame00001.png"]
    assert loaded.pair_split_names is not None
    assert loaded.pair_split_names.tolist() == ["train", "train"]
    assert loaded_metadata["split_counts"] == {"train": 2}
    assert loaded_metadata["pair_type_counts"] == {"A_gt": 2}
    assert loaded_metadata["shards"][0]["query_id"] == "seq8/frame00001.png"
    assert loaded_metadata["shards"][0]["split"] == "train"
    assert loaded_metadata["shards"][0]["pair_type"] == "A_gt"
    assert (tmp_path / "shards" / "shard_00000.npz").exists()


def test_train_matcha_joint_model_from_manifest_supports_lazy_validation_manifest(tmp_path) -> None:
    feature_map = np.eye(8, dtype=np.float32).T.reshape(8, 2, 4)
    supervision = _toy_supervision()
    train_set = build_matcha_joint_index_training_set_from_maps(
        feature_map,
        feature_map.copy(),
        supervision,
        hard_negatives_per_match=1,
    )
    val_set = build_matcha_joint_index_training_set_from_maps(
        feature_map,
        feature_map.copy(),
        supervision,
        hard_negatives_per_match=1,
    )
    object.__setattr__(train_set, "pair_query_ids", np.asarray(["seq1/frame00001.png"], dtype=object))
    object.__setattr__(train_set, "pair_split_names", np.asarray(["train"], dtype=object))
    object.__setattr__(train_set, "pair_type_names", np.asarray(["A_gt"], dtype=object))
    object.__setattr__(val_set, "pair_query_ids", np.asarray(["seq1/frame00033.png"], dtype=object))
    object.__setattr__(val_set, "pair_split_names", np.asarray(["val"], dtype=object))
    object.__setattr__(val_set, "pair_type_names", np.asarray(["A_gt"], dtype=object))
    train_manifest = tmp_path / "train_manifest.json"
    val_manifest = tmp_path / "val_manifest.json"
    save_matcha_joint_training_set_manifest([train_set], train_manifest)
    save_matcha_joint_training_set_manifest([val_set], val_manifest)

    run = train_matcha_joint_model_from_manifest(
        train_manifest,
        MatchaJointTrainingConfig(
            model_type="radio_dual_attention",
            fine_input_dim=4,
            coarse_input_dim=4,
            output_dim=4,
            residual_hidden_dim=8,
            attention_hidden_dim=8,
            attention_depth=1,
            attention_heads=1,
            attention_patch_size=1,
            steps=2,
            batch_size=4,
            lr=1e-3,
            dual_softmax_weight=1.0,
            offset_loss_weight=0.0,
            pair_fine_loss_weight=0.0,
            query_pair_fine_loss_weight=0.0,
            pair_confidence_loss_weight=0.0,
            dense_heatmap_loss_weight=0.0,
            rgb_keypoint_loss_weight=0.0,
            hard_negative_weight=0.0,
            map_pair_batch_size=1,
            device="cpu",
            seed=7,
        ),
        validation_manifest_path=val_manifest,
        validation_interval=1,
        shard_cache_size=1,
        steps_per_shard=1,
    )

    assert run.summary["validation_eval_count"] >= 2
    assert run.summary["validation_manifest_lazy"] is True
    assert run.summary["validation_manifest_shard_count"] == 1
    assert np.isfinite(run.summary["best_validation_loss"])


def test_joint_cache_builder_output_helper_can_write_manifest_without_single_npz(tmp_path) -> None:
    feature_map = np.eye(8, dtype=np.float32).T.reshape(8, 2, 4)
    one = MatchaJointTrainingSet(
        coarse_fine_samples=_toy_samples(),
        query_feature_maps=feature_map[None],
        render_feature_maps=feature_map[None],
        query_heatmap_targets=np.ones((1, 2, 4), dtype=np.float32),
        render_heatmap_targets=np.ones((1, 2, 4), dtype=np.float32),
        query_cell_indices=np.arange(8, dtype=np.int64),
        render_cell_indices=np.arange(8, dtype=np.int64),
    )
    single_output = tmp_path / "single_should_not_exist.npz"
    manifest_output = tmp_path / "joint_manifest.json"

    metadata = _write_matcha_joint_cache_outputs(
        [one, one],
        output=single_output,
        output_manifest=manifest_output,
    )

    assert metadata["joint_cache_manifest"] == str(manifest_output)
    assert "joint_cache" not in metadata
    assert not single_output.exists()
    assert (tmp_path / "shards" / "shard_00001.npz").exists()


def test_train_matcha_joint_model_cli_accepts_joint_cache_manifest(tmp_path) -> None:
    feature_map = np.eye(8, dtype=np.float32).T.reshape(8, 2, 4)
    one = MatchaJointTrainingSet(
        coarse_fine_samples=_toy_samples(),
        query_feature_maps=feature_map[None],
        render_feature_maps=feature_map[None],
        query_heatmap_targets=np.ones((1, 2, 4), dtype=np.float32),
        render_heatmap_targets=np.ones((1, 2, 4), dtype=np.float32),
        query_cell_indices=np.arange(8, dtype=np.int64),
        render_cell_indices=np.arange(8, dtype=np.int64),
    )
    manifest = tmp_path / "joint_manifest.json"
    save_matcha_joint_training_set_manifest([one], manifest)
    adapter_model = tmp_path / "adapter.pt"
    joint_model = tmp_path / "joint.pt"
    summary = tmp_path / "summary.json"

    train_matcha_joint_model_cli(
        [
            "--joint_cache_manifest",
            str(manifest),
            "--output_model",
            str(adapter_model),
            "--output_joint_model",
            str(joint_model),
            "--summary_json",
            str(summary),
            "--output_dim",
            "8",
            "--residual_hidden_dim",
            "16",
            "--steps",
            "1",
            "--batch_size",
            "8",
            "--group_size",
            "4",
            "--device",
            "cpu",
        ]
    )

    assert adapter_model.exists()
    assert joint_model.exists()
    assert summary.exists()


def test_train_matcha_joint_model_cli_supports_lazy_manifest_validation_and_best_checkpoint(tmp_path) -> None:
    feature_map = np.eye(8, dtype=np.float32).T.reshape(8, 2, 4)
    one = MatchaJointTrainingSet(
        coarse_fine_samples=_toy_samples(),
        query_feature_maps=feature_map[None],
        render_feature_maps=feature_map[None],
        query_heatmap_targets=np.ones((1, 2, 4), dtype=np.float32),
        render_heatmap_targets=np.ones((1, 2, 4), dtype=np.float32),
        query_cell_indices=np.arange(8, dtype=np.int64),
        render_cell_indices=np.arange(8, dtype=np.int64),
    )
    manifest = tmp_path / "joint_manifest.json"
    validation_cache = tmp_path / "validation.npz"
    save_matcha_joint_training_set_manifest([one, one], manifest)
    save_matcha_joint_training_set_npz(one, validation_cache)
    adapter_model = tmp_path / "adapter.pt"
    joint_model = tmp_path / "joint.pt"
    best_model = tmp_path / "best_joint.pt"
    summary = tmp_path / "summary.json"

    train_matcha_joint_model_cli(
        [
            "--joint_cache_manifest",
            str(manifest),
            "--lazy_manifest_training",
            "--manifest_shard_cache_size",
            "1",
            "--manifest_steps_per_shard",
            "1",
            "--validation_joint_cache",
            str(validation_cache),
            "--validation_interval",
            "1",
            "--output_model",
            str(adapter_model),
            "--output_joint_model",
            str(joint_model),
            "--output_best_joint_model",
            str(best_model),
            "--summary_json",
            str(summary),
            "--output_dim",
            "8",
            "--residual_hidden_dim",
            "16",
            "--steps",
            "1",
            "--batch_size",
            "8",
            "--group_size",
            "4",
            "--device",
            "cpu",
        ]
    )

    payload = json.loads(summary.read_text())
    assert adapter_model.exists()
    assert joint_model.exists()
    assert best_model.exists()
    assert payload["training"]["manifest_lazy_training"] is True
    assert payload["training"]["best_validation_step"] in {0, 1}
    assert payload["validation_cache"]["sample_count"] == 8


def test_matcha_joint_checkpoint_round_trip_and_projects_full_heads(tmp_path) -> None:
    model = MatchaStyleJointModel(input_dim=8, output_dim=8, residual_hidden_dim=16, group_size=4)
    feature_map = np.eye(8, dtype=np.float32).T.reshape(8, 2, 4)
    run = train_matcha_joint_model(
        MatchaJointTrainingSet(
            coarse_fine_samples=_toy_samples(),
            query_feature_maps=feature_map[None],
            render_feature_maps=feature_map[None],
            query_heatmap_targets=np.ones((1, 2, 4), dtype=np.float32),
            render_heatmap_targets=np.ones((1, 2, 4), dtype=np.float32),
        ),
        MatchaJointTrainingConfig(
            output_dim=8,
            residual_hidden_dim=16,
            steps=5,
            batch_size=8,
            lr=1e-3,
            group_size=4,
            device="cpu",
            seed=9,
        ),
    )
    path = tmp_path / "joint_model.pt"
    save_matcha_joint_model(run, path)

    loaded = load_matcha_joint_model(path, device="cpu")
    selected, offsets, heatmap = project_feature_map_with_matcha_joint_model(loaded.model, feature_map, device="cpu")

    assert loaded.summary["stage"] == "matcha_style_joint_training"
    assert selected.shape == (8, 2, 4)
    assert offsets.shape == (65, 2, 4)
    assert heatmap.shape == (2, 4)
    assert np.all((heatmap >= 0.0) & (heatmap <= 1.0))


def test_map_descriptor_metric_is_computed_per_image_pair() -> None:
    feature_map = np.eye(8, dtype=np.float32).T.reshape(8, 2, 4)
    one = MatchaJointTrainingSet(
        coarse_fine_samples=_toy_samples(),
        query_feature_maps=feature_map[None],
        render_feature_maps=feature_map[None],
        query_heatmap_targets=np.ones((1, 2, 4), dtype=np.float32),
        render_heatmap_targets=np.ones((1, 2, 4), dtype=np.float32),
        query_cell_indices=np.arange(8, dtype=np.int64),
        render_cell_indices=np.arange(8, dtype=np.int64),
    )
    merged = merge_matcha_joint_training_sets([one, one])

    run = train_matcha_joint_model(
        merged,
        MatchaJointTrainingConfig(
            output_dim=8,
            residual_hidden_dim=16,
            steps=20,
            batch_size=8,
            lr=1e-3,
            group_size=4,
            device="cpu",
            seed=12,
        ),
    )

    assert run.summary["map_descriptor_top1_acc"] >= 0.99


def test_large_joint_training_uses_bounded_eval_subset(monkeypatch) -> None:
    feature_map = np.eye(8, dtype=np.float32).T.reshape(8, 2, 4)
    one = MatchaJointTrainingSet(
        coarse_fine_samples=_toy_samples(),
        query_feature_maps=feature_map[None],
        render_feature_maps=feature_map[None],
        query_heatmap_targets=np.ones((1, 2, 4), dtype=np.float32),
        render_heatmap_targets=np.ones((1, 2, 4), dtype=np.float32),
        query_cell_indices=np.arange(8, dtype=np.int64),
        render_cell_indices=np.arange(8, dtype=np.int64),
    )
    merged = merge_matcha_joint_training_sets([one, one, one, one])
    original = joint_training._dual_softmax_descriptor_loss_and_confidence
    observed_batch_sizes: list[int] = []

    def checked_dual_softmax(query_z, render_z, *args, **kwargs):
        observed_batch_sizes.append(int(query_z.shape[0]))
        assert int(query_z.shape[0]) <= 8
        assert int(render_z.shape[0]) <= 8
        return original(query_z, render_z, *args, **kwargs)

    monkeypatch.setattr(joint_training, "_dual_softmax_descriptor_loss_and_confidence", checked_dual_softmax)

    run = train_matcha_joint_model(
        merged,
        MatchaJointTrainingConfig(
            output_dim=8,
            residual_hidden_dim=16,
            steps=2,
            batch_size=8,
            lr=1e-3,
            group_size=4,
            device="cpu",
            seed=21,
        ),
    )

    assert observed_batch_sizes
    assert run.summary["eval_sample_count"] == 8


def test_original_fine_matcher_outputs_64_coordinate_bins() -> None:
    matcher = joint_training._OriginalMatchaFineMatcher(descriptor_dim=8, hidden_dim=16)
    matcher.train()

    logits = matcher(torch.randn(4, 8), torch.randn(4, 8))

    assert logits.shape == (4, 64)


def test_pair_fine_losses_ignore_dustbin_labels_and_report_valid_metrics() -> None:
    class _FineOnlyModel(torch.nn.Module):
        def forward_rows(self, features):
            return features, torch.zeros(features.shape[0], 65, device=features.device)

        def pair_fine_logits(self, query_z, render_z):
            logits = torch.full((query_z.shape[0], 64), -10.0, device=query_z.device)
            logits[0, 1] = 10.0
            logits[1, 0] = 10.0
            logits[2, 2] = 10.0
            return logits

        def query_pair_fine_logits(self, query_z, render_z):
            logits = torch.full((query_z.shape[0], 64), -10.0, device=query_z.device)
            logits[0, 3] = 10.0
            logits[1, 0] = 10.0
            logits[2, 4] = 10.0
            return logits

    features = np.eye(3, dtype=np.float32)
    samples = MatchaJointTrainingSet(
        coarse_fine_samples=MatchaCoarseFineTrainingSet(
            query_features=features,
            render_features=features,
            query_offset_labels=np.asarray([3, 64, 4], dtype=np.int64),
            render_offset_labels=np.asarray([1, 64, 2], dtype=np.int64),
            negative_render_features=np.zeros((3, 0, 3), dtype=np.float32),
            roundtrip_errors_px=np.zeros((3,), dtype=np.float32),
        )
    )

    loss, metrics = joint_training._total_loss(
        _FineOnlyModel(),
        samples,
        np.arange(3, dtype=np.int64),
        MatchaJointTrainingConfig(
            output_dim=3,
            steps=1,
            batch_size=3,
            dual_softmax_weight=0.0,
            offset_loss_weight=0.0,
            pair_fine_loss_weight=1.0,
            query_pair_fine_loss_weight=1.0,
            pair_confidence_loss_weight=0.0,
            hard_negative_weight=0.0,
            device="cpu",
        ),
        torch.device("cpu"),
        seed=1,
    )

    assert torch.isfinite(loss)
    assert metrics["render_pair_fine_valid_count"] == 2.0
    assert metrics["query_pair_fine_valid_count"] == 2.0
    assert metrics["render_pair_fine_acc"] == 1.0
    assert metrics["query_pair_fine_acc"] == 1.0


def test_full_map_auxiliary_losses_use_pair_batches(monkeypatch) -> None:
    feature_map = np.eye(8, dtype=np.float32).T.reshape(8, 2, 4)
    one = MatchaJointTrainingSet(
        coarse_fine_samples=_toy_samples(),
        query_feature_maps=feature_map[None],
        render_feature_maps=feature_map[None],
        query_heatmap_targets=np.ones((1, 2, 4), dtype=np.float32),
        render_heatmap_targets=np.ones((1, 2, 4), dtype=np.float32),
        query_cell_indices=np.arange(8, dtype=np.int64),
        render_cell_indices=np.arange(8, dtype=np.int64),
    )
    merged = merge_matcha_joint_training_sets([one, one, one, one])
    original = joint_training.MatchaStyleJointModel.forward_feature_map
    observed_map_batches: list[int] = []

    def checked_forward_feature_map(self, feature_maps):
        observed_map_batches.append(int(feature_maps.shape[0]))
        assert int(feature_maps.shape[0]) <= 2
        return original(self, feature_maps)

    monkeypatch.setattr(joint_training.MatchaStyleJointModel, "forward_feature_map", checked_forward_feature_map)

    train_matcha_joint_model(
        merged,
        MatchaJointTrainingConfig(
            output_dim=8,
            residual_hidden_dim=16,
            steps=2,
            batch_size=16,
            map_pair_batch_size=2,
            lr=1e-3,
            dense_heatmap_loss_weight=0.25,
            local_fine_transformer_loss_weight=0.25,
            group_size=4,
            device="cpu",
            seed=22,
        ),
    )

    assert observed_map_batches


def test_full_joint_cache_builder_creates_heatmaps_indices_and_rgb_labels() -> None:
    query_map = np.eye(8, dtype=np.float32).T.reshape(8, 2, 4)
    render_map = query_map.copy()
    rgb = np.zeros((16, 32, 3), dtype=np.uint8)
    query_labels = np.full((2, 4), 64, dtype=np.int64)
    render_labels = np.full((2, 4), 64, dtype=np.int64)
    query_labels[0, 0] = 36
    render_labels[0, 0] = 18

    qheat, rheat = heatmap_targets_from_coarse_supervision(
        _toy_supervision(),
        query_grid_hw=(2, 4),
        render_grid_hw=(2, 4),
        roundtrip_threshold_px=2.0,
    )
    assert qheat.shape == (2, 4)
    assert rheat.shape == (2, 4)
    assert qheat[0, 0] == 1.0
    assert qheat[0, 3] == 0.0

    samples = build_matcha_joint_training_set_from_maps(
        query_map,
        render_map,
        _toy_supervision(),
        query_rgb=rgb,
        render_rgb=rgb,
        query_keypoint_label_map=query_labels,
        render_keypoint_label_map=render_labels,
        hard_negatives_per_match=2,
        roundtrip_heatmap_threshold_px=2.0,
    )

    assert samples.coarse_fine_samples.sample_count == 8
    assert samples.coarse_fine_samples.metadata["positive_match_count"] == 4
    assert samples.coarse_fine_samples.metadata["no_match_count"] == 4
    assert samples.sample_no_match_labels is not None
    assert samples.sample_no_match_labels.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert samples.query_feature_maps is not None
    assert samples.query_feature_maps.shape == (1, 8, 2, 4)
    assert samples.query_heatmap_targets is not None
    assert samples.query_heatmap_targets[0, 0, 0] == 1.0
    assert samples.query_cell_indices is not None
    assert samples.query_cell_indices.tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
    assert samples.query_rgb_images is not None
    assert samples.query_rgb_images.shape == (1, 3, 16, 32)
    assert samples.query_rgb_keypoint_labels is not None
    assert samples.query_rgb_keypoint_labels[0, 0, 0] == 36
