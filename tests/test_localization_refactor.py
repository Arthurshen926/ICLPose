from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from feature_extract.vfm.localization import (
    AdapterFeatureMapper,
    CoarseProposal,
    FeatureMapPair,
    JointFeatureMapper,
    MatchaCoarseMatcher,
    MatchaTopKCoarseMatcher,
    RGBPatchMeasurementAdapter,
    SelectorCoarseMeasurementModel,
)
from feature_extract.vfm.matcha_coarse_fine_adapter import MatchaCoarseFineAdapter
from feature_extract.vfm.matcha_joint_training import RadioDualAttentionFusionJointModel


def test_feature_map_pair_validates_real_query_and_reference_maps() -> None:
    query = np.zeros((4, 2, 3), dtype=np.float32)
    reference = np.zeros((4, 2, 3), dtype=np.float32)

    pair = FeatureMapPair(
        query=query,
        reference=reference,
        query_image_size=(30, 20),
        reference_image_size=(30, 20),
    )

    assert pair.channels == 4
    assert pair.query_grid_hw == (2, 3)
    assert pair.reference_grid_hw == (2, 3)


def test_coarse_proposal_keeps_match_metadata_without_render_fields() -> None:
    proposal = CoarseProposal(
        query_index=1,
        reference_index=2,
        query_xy=np.asarray([10.0, 20.0], dtype=np.float32),
        reference_xy=np.asarray([30.0, 40.0], dtype=np.float32),
        score=0.7,
        confidence=0.6,
        rank=0,
    )

    assert proposal.query_index == 1
    assert proposal.reference_index == 2
    assert proposal.confidence == 0.6


def test_adapter_feature_mapper_projects_raw_radio_maps_to_descriptor_maps() -> None:
    model = MatchaCoarseFineAdapter(input_dim=4, output_dim=4, residual_hidden_dim=8, group_size=2)
    mapper = AdapterFeatureMapper(model, device="cpu")
    raw = np.eye(4, dtype=np.float32).T.reshape(4, 2, 2)

    output = mapper.project(raw)

    assert output.coarse_descriptors.shape == (4, 2, 2)
    assert output.measurement_context.shape == (4, 2, 2)
    assert output.offset_logits is not None
    assert output.offset_logits.shape == (65, 2, 2)
    assert torch.isfinite(torch.from_numpy(output.coarse_descriptors)).all()


def test_joint_feature_mapper_exposes_coarse_and_measurement_context_separately() -> None:
    model = RadioDualAttentionFusionJointModel(
        fine_input_dim=4,
        coarse_input_dim=4,
        output_dim=4,
        residual_hidden_dim=8,
        attention_hidden_dim=8,
        attention_depth=1,
        attention_heads=1,
        attention_patch_size=1,
        attention_fusion_mode="matcha_original",
        group_size=4,
    )
    model.eval()
    mapper = JointFeatureMapper(model, device="cpu")
    fine = np.eye(4, dtype=np.float32).T.reshape(1, 4, 2, 2)
    coarse = np.flip(fine, axis=1).copy()
    raw = np.concatenate([fine, coarse], axis=1)[0]

    output = mapper.project(raw)
    with torch.no_grad():
        coarse_desc, fine_desc, _heat = model.forward_fuse_feature(torch.as_tensor(raw[None], dtype=torch.float32))

    np.testing.assert_allclose(output.coarse_descriptors, coarse_desc[0].numpy(), atol=1e-6)
    np.testing.assert_allclose(output.measurement_context, fine_desc[0].numpy(), atol=1e-6)
    assert output.offset_logits is not None
    assert output.offset_logits.shape == (65, 2, 2)
    assert output.heatmap is not None
    assert output.heatmap.shape == (2, 2)


def test_matcha_coarse_matcher_returns_reference_proposals_without_fine_measurement() -> None:
    query = np.zeros((2, 1, 2), dtype=np.float32)
    reference = np.zeros((2, 1, 2), dtype=np.float32)
    query[:, 0, 0] = [1.0, 0.0]
    query[:, 0, 1] = [0.0, 1.0]
    reference[:, 0, 0] = [1.0, 0.0]
    reference[:, 0, 1] = [0.0, 1.0]
    matcher = MatchaCoarseMatcher(logit_scale=12.0, mutual=True)

    proposals = matcher.match(
        query,
        reference,
        query_image_size=(20, 10),
        reference_image_size=(20, 10),
    )

    assert [(item.query_index, item.reference_index) for item in proposals] == [(0, 0), (1, 1)]
    assert all(item.confidence is not None for item in proposals)


def test_matcha_topk_coarse_matcher_returns_ranked_reference_proposals() -> None:
    query = np.zeros((2, 1, 1), dtype=np.float32)
    reference = np.zeros((2, 1, 3), dtype=np.float32)
    query[:, 0, 0] = [1.0, 0.0]
    reference[:, 0, 0] = [0.8, 0.2]
    reference[:, 0, 1] = [1.0, 0.0]
    reference[:, 0, 2] = [0.7, 0.3]
    matcher = MatchaTopKCoarseMatcher(k_per_query=2, mutual_mode="annotate", logit_scale=10.0)

    proposals = matcher.match(
        query,
        reference,
        query_image_size=(10, 10),
        reference_image_size=(30, 10),
    )

    assert [(item.query_index, item.reference_index) for item in proposals] == [(0, 1), (0, 0)]
    assert [item.rank for item in proposals] == [0, 1]
    assert proposals[0].score > proposals[1].score
    assert proposals[0].metadata["coarse_score_gap"] == 0.0


def test_matcha_topk_coarse_matcher_accepts_none_mutual_mode() -> None:
    query = np.zeros((2, 1, 1), dtype=np.float32)
    reference = np.zeros((2, 1, 1), dtype=np.float32)
    query[:, 0, 0] = [1.0, 0.0]
    reference[:, 0, 0] = [1.0, 0.0]
    matcher = MatchaTopKCoarseMatcher(k_per_query=1, mutual_mode=None, logit_scale=10.0)

    proposals = matcher.match(
        query,
        reference,
        query_image_size=(10, 10),
        reference_image_size=(10, 10),
    )

    assert len(proposals) == 1


def test_rgb_patch_measurement_adapter_returns_measurements_for_reference_proposals() -> None:
    class FakeBranch:
        crop_radius_px = 2.0
        step_px = 1.0

        def __init__(self) -> None:
            self.query_patch_shape = None
            self.reference_patch_shape = None

        def to(self, _device):
            return self

        def eval(self):
            return self

        def forward_from_patches(self, query_patch, render_patch, prior_scale_px=None):
            self.query_patch_shape = tuple(query_patch.shape)
            self.reference_patch_shape = tuple(render_patch.shape)
            return SimpleNamespace(
                gated_mean_offset_xy=torch.asarray([[1.5, -0.5]], dtype=torch.float32),
                mean_offset_xy=torch.asarray([[0.0, 0.0]], dtype=torch.float32),
                direct_mean_offset_xy=torch.asarray([[0.0, 0.0]], dtype=torch.float32),
                direct_log_sigma_xy=torch.zeros((1, 2), dtype=torch.float32),
                dustbin_logit=torch.asarray([-2.0], dtype=torch.float32),
            )

    branch = FakeBranch()
    adapter = RGBPatchMeasurementAdapter(branch=branch, device="cpu", prediction_head="gated")
    proposal = CoarseProposal(
        query_index=0,
        reference_index=0,
        query_xy=np.asarray([4.0, 5.0], dtype=np.float32),
        reference_xy=np.asarray([6.0, 7.0], dtype=np.float32),
        score=1.0,
    )
    query_rgb = np.zeros((3, 12, 12), dtype=np.float32)
    reference_rgb = np.zeros((3, 14, 14), dtype=np.float32)

    measurements = adapter.measure(query_rgb, reference_rgb, [proposal])

    assert len(measurements) == 1
    np.testing.assert_allclose(measurements[0].measured_query_xy, [5.5, 4.5], atol=1e-6)
    np.testing.assert_allclose(measurements[0].measured_reference_xy, [6.0, 7.0], atol=1e-6)
    assert measurements[0].confidence is not None
    assert measurements[0].confidence > 0.8
    assert branch.query_patch_shape is not None
    assert branch.reference_patch_shape is not None


def test_rgb_patch_measurement_adapter_batches_large_proposal_sets() -> None:
    class FakeBranch:
        crop_radius_px = 1.0
        step_px = 1.0

        def __init__(self) -> None:
            self.batch_shapes = []

        def to(self, _device):
            return self

        def eval(self):
            return self

        def forward_from_patches(self, query_patch, render_patch, prior_scale_px=None):
            del render_patch, prior_scale_px
            self.batch_shapes.append(tuple(query_patch.shape))
            batch = int(query_patch.shape[0])
            return SimpleNamespace(
                gated_mean_offset_xy=torch.zeros((batch, 2), dtype=torch.float32),
                mean_offset_xy=torch.zeros((batch, 2), dtype=torch.float32),
                direct_mean_offset_xy=torch.zeros((batch, 2), dtype=torch.float32),
                direct_log_sigma_xy=torch.zeros((batch, 2), dtype=torch.float32),
                dustbin_logit=torch.zeros((batch,), dtype=torch.float32),
            )

    proposals = [
        CoarseProposal(
            query_index=index,
            reference_index=index,
            query_xy=np.asarray([4.0, 4.0], dtype=np.float32),
            reference_xy=np.asarray([5.0, 5.0], dtype=np.float32),
            score=1.0,
        )
        for index in range(5)
    ]
    branch = FakeBranch()
    adapter = RGBPatchMeasurementAdapter(branch=branch, device="cpu", batch_size=2)

    measurements = adapter.measure(
        np.zeros((3, 12, 12), dtype=np.float32),
        np.zeros((3, 12, 12), dtype=np.float32),
        proposals,
    )

    assert len(measurements) == 5
    assert [shape[0] for shape in branch.batch_shapes] == [2, 2, 1]


def test_rgb_patch_measurement_adapter_batches_across_reference_images() -> None:
    class FakeBranch:
        crop_radius_px = 1.0
        step_px = 1.0

        def __init__(self) -> None:
            self.batch_shapes = []

        def to(self, _device):
            return self

        def eval(self):
            return self

        def forward_from_patches(self, query_patch, render_patch, prior_scale_px=None):
            del render_patch, prior_scale_px
            self.batch_shapes.append(tuple(query_patch.shape))
            batch = int(query_patch.shape[0])
            return SimpleNamespace(
                gated_mean_offset_xy=torch.zeros((batch, 2), dtype=torch.float32),
                mean_offset_xy=torch.zeros((batch, 2), dtype=torch.float32),
                direct_mean_offset_xy=torch.zeros((batch, 2), dtype=torch.float32),
                direct_log_sigma_xy=torch.zeros((batch, 2), dtype=torch.float32),
                dustbin_logit=torch.zeros((batch,), dtype=torch.float32),
            )

    def proposal(index: int) -> CoarseProposal:
        return CoarseProposal(
            query_index=index,
            reference_index=index,
            query_xy=np.asarray([4.0 + index, 4.0], dtype=np.float32),
            reference_xy=np.asarray([5.0, 5.0], dtype=np.float32),
            score=1.0,
        )

    branch = FakeBranch()
    adapter = RGBPatchMeasurementAdapter(branch=branch, device="cpu", batch_size=3)

    measurements = adapter.measure_by_reference(
        np.zeros((3, 12, 12), dtype=np.float32),
        {
            "r0.png": np.zeros((3, 12, 12), dtype=np.float32),
            "r1.png": np.zeros((3, 12, 12), dtype=np.float32),
        },
        {
            "r0.png": [proposal(0), proposal(1)],
            "r1.png": [proposal(2), proposal(3), proposal(4)],
        },
    )

    assert set(measurements) == {"r0.png", "r1.png"}
    assert len(measurements["r0.png"]) == 2
    assert len(measurements["r1.png"]) == 3
    assert [shape[0] for shape in branch.batch_shapes] == [3, 2]


def test_rgb_patch_measurement_adapter_calibrates_confidence_and_uncertainty() -> None:
    class FakeBranch:
        crop_radius_px = 1.0
        step_px = 1.0

        def to(self, _device):
            return self

        def eval(self):
            return self

        def forward_from_patches(self, query_patch, render_patch, prior_scale_px=None):
            del query_patch, render_patch, prior_scale_px
            return SimpleNamespace(
                gated_mean_offset_xy=torch.zeros((2, 2), dtype=torch.float32),
                mean_offset_xy=torch.zeros((2, 2), dtype=torch.float32),
                direct_mean_offset_xy=torch.zeros((2, 2), dtype=torch.float32),
                direct_log_sigma_xy=torch.log(torch.asarray([[2.0, 4.0], [1.0, 3.0]], dtype=torch.float32)),
                dustbin_logit=torch.asarray([-8.0, -2.0], dtype=torch.float32),
            )

    proposals = [
        CoarseProposal(
            query_index=index,
            reference_index=index,
            query_xy=np.asarray([4.0, 4.0], dtype=np.float32),
            reference_xy=np.asarray([5.0, 5.0], dtype=np.float32),
            score=1.0,
        )
        for index in range(2)
    ]
    adapter = RGBPatchMeasurementAdapter(
        branch=FakeBranch(),
        device="cpu",
        batch_size=2,
        confidence_temperature=4.0,
        uncertainty_scale=2.0,
        uncertainty_floor_px=0.5,
    )

    measurements = adapter.measure(
        np.zeros((3, 12, 12), dtype=np.float32),
        np.zeros((3, 12, 12), dtype=np.float32),
        proposals,
    )

    assert measurements[0].confidence == pytest.approx(float(torch.sigmoid(torch.tensor(2.0)).item()))
    assert measurements[1].confidence == pytest.approx(float(torch.sigmoid(torch.tensor(0.5)).item()))
    assert measurements[0].confidence < 0.99
    assert measurements[0].uncertainty_px == pytest.approx(6.5)
    assert measurements[1].uncertainty_px == pytest.approx(4.5)


def test_rgb_patch_measurement_adapter_batches_across_queries() -> None:
    class FakeBranch:
        crop_radius_px = 1.0
        step_px = 1.0

        def __init__(self) -> None:
            self.batch_shapes = []

        def to(self, _device):
            return self

        def eval(self):
            return self

        def forward_from_patches(self, query_patch, render_patch, prior_scale_px=None):
            del render_patch, prior_scale_px
            self.batch_shapes.append(tuple(query_patch.shape))
            batch = int(query_patch.shape[0])
            return SimpleNamespace(
                gated_mean_offset_xy=torch.zeros((batch, 2), dtype=torch.float32),
                mean_offset_xy=torch.zeros((batch, 2), dtype=torch.float32),
                direct_mean_offset_xy=torch.zeros((batch, 2), dtype=torch.float32),
                direct_log_sigma_xy=torch.zeros((batch, 2), dtype=torch.float32),
                dustbin_logit=torch.zeros((batch,), dtype=torch.float32),
            )

    def proposal(index: int) -> CoarseProposal:
        return CoarseProposal(
            query_index=index,
            reference_index=index,
            query_xy=np.asarray([4.0 + index, 4.0], dtype=np.float32),
            reference_xy=np.asarray([5.0, 5.0], dtype=np.float32),
            score=1.0,
        )

    branch = FakeBranch()
    adapter = RGBPatchMeasurementAdapter(branch=branch, device="cpu", batch_size=4)

    measurements = adapter.measure_many_by_reference(
        {
            "q0.png": np.zeros((3, 12, 12), dtype=np.float32),
            "q1.png": np.zeros((3, 12, 12), dtype=np.float32),
        },
        {
            "r0.png": np.zeros((3, 12, 12), dtype=np.float32),
            "r1.png": np.zeros((3, 12, 12), dtype=np.float32),
        },
        {
            ("q0.png", "r0.png"): [proposal(0), proposal(1)],
            ("q1.png", "r0.png"): [proposal(2)],
            ("q1.png", "r1.png"): [proposal(3), proposal(4)],
        },
    )

    assert set(measurements) == {("q0.png", "r0.png"), ("q1.png", "r0.png"), ("q1.png", "r1.png")}
    assert len(measurements[("q0.png", "r0.png")]) == 2
    assert len(measurements[("q1.png", "r0.png")]) == 1
    assert len(measurements[("q1.png", "r1.png")]) == 2
    assert [shape[0] for shape in branch.batch_shapes] == [4, 1]


def test_selector_coarse_measurement_model_runs_selector_and_coarse_matcher() -> None:
    adapter = MatchaCoarseFineAdapter(input_dim=4, output_dim=4, residual_hidden_dim=8, group_size=2)
    model = SelectorCoarseMeasurementModel(
        feature_mapper=AdapterFeatureMapper(adapter, device="cpu"),
        coarse_matcher=MatchaCoarseMatcher(logit_scale=12.0, mutual=True),
    )
    raw = np.eye(4, dtype=np.float32).T.reshape(4, 2, 2)

    result = model.match_pair(
        raw,
        raw.copy(),
        query_image_size=(20, 20),
        reference_image_size=(20, 20),
    )

    assert result.mapped_query.coarse_descriptors.shape == (4, 2, 2)
    assert len(result.coarse_proposals) == 4
    assert result.measurements == []


def test_selector_coarse_measurement_model_passes_mapped_context_to_measurement_branch() -> None:
    class ContextCheckingMeasurement:
        def __init__(self) -> None:
            self.seen_context = False

        def measure(self, query_rgb, reference_rgb, proposals, *, mapped_query=None, mapped_reference=None):
            self.seen_context = mapped_query is not None and mapped_reference is not None
            return [
                SimpleNamespace(
                    proposal=proposals[0],
                    measured_query_xy=proposals[0].query_xy,
                    measured_reference_xy=proposals[0].reference_xy,
                    confidence=1.0,
                    uncertainty_px=None,
                )
            ]

    adapter = MatchaCoarseFineAdapter(input_dim=4, output_dim=4, residual_hidden_dim=8, group_size=2)
    measurement = ContextCheckingMeasurement()
    model = SelectorCoarseMeasurementModel(
        feature_mapper=AdapterFeatureMapper(adapter, device="cpu"),
        coarse_matcher=MatchaCoarseMatcher(logit_scale=12.0, mutual=True, max_matches=1),
        measurement_branch=measurement,
    )
    raw = np.eye(4, dtype=np.float32).T.reshape(4, 2, 2)

    result = model.match_pair(
        raw,
        raw.copy(),
        query_image_size=(20, 20),
        reference_image_size=(20, 20),
        query_rgb=np.zeros((3, 20, 20), dtype=np.float32),
        reference_rgb=np.zeros((3, 20, 20), dtype=np.float32),
    )

    assert measurement.seen_context is True
    assert len(result.measurements) == 1
