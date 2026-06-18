from __future__ import annotations

import numpy as np
import torch

from feature_extract.tools.vfm.eval_matcha_2dgs_synthetic_pairs import (
    _matcha_eval_matches_for_pnp,
    _refine_matches_with_pair_conditioned_patch_corr,
    _refine_matches_with_pair_conditioned_local_window,
)
from feature_extract.vfm.rendered_keypoint_matching import KeypointFeatureMatch


class _FixedLocalWindowModel:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], tuple[int, ...], list[int], list[int]]] = []

    def to(self, _device):
        return self

    def eval(self):
        return self

    def local_window_fine_logits_from_maps(self, qmaps, rmaps, pairs, qidx, ridx):
        self.calls.append(
            (
                tuple(qmaps.shape),
                tuple(rmaps.shape),
                [int(item) for item in qidx.detach().cpu().tolist()],
                [int(item) for item in ridx.detach().cpu().tolist()],
            )
        )
        logits = torch.full((int(qidx.numel()), 64), -20.0, dtype=torch.float32, device=qmaps.device)
        label = 63 if len(self.calls) == 1 else 0
        logits[:, label] = 20.0
        return logits


class _FixedPatchCorrModel:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], tuple[int, ...], list[int], list[int], list[list[float]]]] = []

    def to(self, _device):
        return self

    def eval(self):
        return self

    def patch_corr_fine_logits_from_maps_and_rgb(
        self,
        qmaps,
        rmaps,
        _qrgb,
        _rrgb,
        _pairs,
        qidx,
        ridx,
        *,
        query_xy,
    ):
        self.calls.append(
            (
                tuple(qmaps.shape),
                tuple(rmaps.shape),
                [int(item) for item in qidx.detach().cpu().tolist()],
                [int(item) for item in ridx.detach().cpu().tolist()],
                [[float(v) for v in row] for row in query_xy.detach().cpu().tolist()],
            )
        )
        logits = torch.full((int(qidx.numel()), 64), -20.0, dtype=torch.float32, device=qmaps.device)
        label = 63 if len(self.calls) == 1 else 0
        logits[:, label] = 20.0
        return logits


class _ConfidencePatchCorrModel(_FixedPatchCorrModel):
    def patch_corr_fine_logits_from_maps_and_rgb(
        self,
        qmaps,
        rmaps,
        _qrgb,
        _rrgb,
        _pairs,
        qidx,
        ridx,
        *,
        query_xy,
    ):
        self.calls.append(
            (
                tuple(qmaps.shape),
                tuple(rmaps.shape),
                [int(item) for item in qidx.detach().cpu().tolist()],
                [int(item) for item in ridx.detach().cpu().tolist()],
                [[float(v) for v in row] for row in query_xy.detach().cpu().tolist()],
            )
        )
        logits = torch.zeros((int(qidx.numel()), 64), dtype=torch.float32, device=qmaps.device)
        for row, (source_idx, target_idx) in enumerate(zip(qidx.detach().cpu().tolist(), ridx.detach().cpu().tolist())):
            if int(source_idx) == 0 or int(target_idx) == 0:
                logits[row, 63] = 20.0
        return logits


def test_pair_conditioned_local_window_refines_query_and_render_sides() -> None:
    model = _FixedLocalWindowModel()
    match = KeypointFeatureMatch(
        query_index=0,
        render_index=0,
        query_xy=np.asarray([4.0, 4.0], dtype=np.float64),
        render_xy=np.asarray([4.0, 4.0], dtype=np.float64),
        similarity=0.9,
        ratio=1.0,
    )

    refined = _refine_matches_with_pair_conditioned_local_window(
        [match],
        model,
        np.zeros((2, 2, 2), dtype=np.float32),
        np.zeros((2, 2, 2), dtype=np.float32),
        query_image_width=16,
        query_image_height=16,
        render_image_width=16,
        render_image_height=16,
        device="cpu",
    )

    assert len(refined) == 1
    assert np.allclose(refined[0].render_xy, [7.5, 7.5])
    assert np.allclose(refined[0].query_xy, [0.5, 0.5])
    assert len(model.calls) == 2
    assert model.calls[0][2:] == ([0], [0])
    assert model.calls[1][2:] == ([0], [0])


def test_pair_conditioned_patch_corr_refines_query_and_render_sides() -> None:
    model = _FixedPatchCorrModel()
    match = KeypointFeatureMatch(
        query_index=0,
        render_index=0,
        query_xy=np.asarray([4.0, 4.0], dtype=np.float64),
        render_xy=np.asarray([4.0, 4.0], dtype=np.float64),
        similarity=0.9,
        ratio=1.0,
    )
    image = np.zeros((16, 16, 3), dtype=np.uint8)

    refined = _refine_matches_with_pair_conditioned_patch_corr(
        [match],
        model,
        np.zeros((2, 2, 2), dtype=np.float32),
        np.zeros((2, 2, 2), dtype=np.float32),
        image,
        image,
        query_image_width=16,
        query_image_height=16,
        render_image_width=16,
        render_image_height=16,
        device="cpu",
    )

    assert len(refined) == 1
    assert np.allclose(refined[0].render_xy, [7.5, 7.5])
    assert np.allclose(refined[0].query_xy, [0.5, 0.5])
    assert len(model.calls) == 2
    assert model.calls[0][2:4] == ([0], [0])
    assert model.calls[1][2:4] == ([0], [0])
    assert np.allclose(model.calls[0][4], [[4.0, 4.0]])
    assert np.allclose(model.calls[1][4], [[7.5, 7.5]])


def test_pair_conditioned_patch_corr_can_blend_fine_confidence_into_order() -> None:
    model = _ConfidencePatchCorrModel()
    low_fine = KeypointFeatureMatch(
        query_index=1,
        render_index=1,
        query_xy=np.asarray([4.0, 4.0], dtype=np.float64),
        render_xy=np.asarray([4.0, 4.0], dtype=np.float64),
        similarity=0.9,
        ratio=1.0,
        dual_softmax_confidence=0.99,
    )
    high_fine = KeypointFeatureMatch(
        query_index=0,
        render_index=0,
        query_xy=np.asarray([4.0, 4.0], dtype=np.float64),
        render_xy=np.asarray([4.0, 4.0], dtype=np.float64),
        similarity=0.1,
        ratio=1.0,
        dual_softmax_confidence=0.01,
    )
    image = np.zeros((16, 16, 3), dtype=np.uint8)

    refined = _refine_matches_with_pair_conditioned_patch_corr(
        [low_fine, high_fine],
        model,
        np.zeros((2, 2, 2), dtype=np.float32),
        np.zeros((2, 2, 2), dtype=np.float32),
        image,
        image,
        query_image_width=16,
        query_image_height=16,
        render_image_width=16,
        render_image_height=16,
        device="cpu",
        fine_confidence_blend=1.0,
    )

    assert [item.query_index for item in refined] == [0, 1]
    assert float(refined[0].dual_softmax_confidence or 0.0) > float(refined[1].dual_softmax_confidence or 0.0)


def test_matcha_eval_matches_use_pair_conditioned_refinement_before_dense_offsets() -> None:
    model = _FixedLocalWindowModel()
    query = np.zeros((2, 1, 1), dtype=np.float32)
    render = np.zeros((2, 1, 1), dtype=np.float32)
    query[:, 0, 0] = [1.0, 0.0]
    render[:, 0, 0] = [1.0, 0.0]
    dense_query_offsets = np.full((65, 1, 1), -20.0, dtype=np.float32)
    dense_render_offsets = np.full((65, 1, 1), -20.0, dtype=np.float32)
    dense_query_offsets[10, 0, 0] = 20.0
    dense_render_offsets[10, 0, 0] = 20.0

    matches, refinement_source, fallback_reason = _matcha_eval_matches_for_pnp(
        query,
        render,
        query_image_width=16,
        query_image_height=16,
        render_image_width=16,
        render_image_height=16,
        query_offset_logits=dense_query_offsets,
        render_offset_logits=dense_render_offsets,
        joint_model=model,
        raw_query_feature_map=query,
        raw_render_feature_map=render,
        device="cpu",
    )

    assert refinement_source == "pair_conditioned_local_window"
    assert fallback_reason is None
    assert len(matches) == 1
    assert np.allclose(matches[0].render_xy, [15.0, 15.0])
    assert np.allclose(matches[0].query_xy, [1.0, 1.0])


def test_matcha_eval_matches_can_use_pair_conditioned_patch_corr_before_dense_offsets() -> None:
    model = _FixedPatchCorrModel()
    query = np.zeros((2, 1, 1), dtype=np.float32)
    render = np.zeros((2, 1, 1), dtype=np.float32)
    query[:, 0, 0] = [1.0, 0.0]
    render[:, 0, 0] = [1.0, 0.0]
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    dense_query_offsets = np.full((65, 1, 1), -20.0, dtype=np.float32)
    dense_render_offsets = np.full((65, 1, 1), -20.0, dtype=np.float32)
    dense_query_offsets[10, 0, 0] = 20.0
    dense_render_offsets[10, 0, 0] = 20.0

    matches, refinement_source, fallback_reason = _matcha_eval_matches_for_pnp(
        query,
        render,
        query_image_width=8,
        query_image_height=8,
        render_image_width=8,
        render_image_height=8,
        query_offset_logits=dense_query_offsets,
        render_offset_logits=dense_render_offsets,
        joint_model=model,
        raw_query_feature_map=query,
        raw_render_feature_map=render,
        query_rgb=image,
        render_rgb=image,
        fine_refinement_mode="patch_corr",
        device="cpu",
    )

    assert refinement_source == "pair_conditioned_patch_corr"
    assert fallback_reason is None
    assert len(matches) == 1
    assert np.allclose(matches[0].render_xy, [7.5, 7.5])
    assert np.allclose(matches[0].query_xy, [0.5, 0.5])
