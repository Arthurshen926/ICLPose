from __future__ import annotations

import inspect

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.matcha_multiview_supervision import (
    MatchaGeometryView,
    MatchaMultiviewSupervisionConfig,
    build_matcha_3dgs_multiview_coarse_supervision,
)
from feature_extract.vfm.matcha_coarse_supervision import MatchaCoarseSupervision, merge_fine_labels_by_cell_pair
from feature_extract.tools.vfm.train_matcha_joint_streaming_model import (
    _label_entropy_bits,
    _render_subcell_seed_xy,
)


def _camera(width: int = 16, height: int = 16) -> ColmapCamera:
    return ColmapCamera(camera_id=1, model_id=1, width=width, height=height, params=(10.0, 10.0, width / 2, height / 2))


def _view(*, x_offset: float = 0.0, depth_value: float = 4.0, alpha_value: float = 1.0) -> MatchaGeometryView:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = float(x_offset)
    return MatchaGeometryView(
        camera=_camera(),
        pose_w2c=pose,
        depth=np.full((16, 16), float(depth_value), dtype=np.float32),
        alpha=np.full((16, 16), float(alpha_value), dtype=np.float32),
        view_id=f"x={x_offset}",
    )


def test_3dgs_multiview_supervision_aggregates_support_views_from_geometry_only() -> None:
    supervision = build_matcha_3dgs_multiview_coarse_supervision(
        query_view=_view(),
        render_view=_view(),
        support_views=(_view(), _view()),
        query_grid_hw=(2, 2),
        render_grid_hw=(2, 2),
        config=MatchaMultiviewSupervisionConfig(min_support_views=2, support_depth_tolerance_m=0.05),
    )

    assert supervision.source == "geometry_3dgs_multiview"
    assert supervision.count == 4
    assert supervision.query_indices.tolist() == [0, 1, 2, 3]
    assert supervision.render_indices.tolist() == [0, 1, 2, 3]
    assert np.allclose(supervision.query_xy, supervision.render_xy)
    assert supervision.query_offset_labels.tolist() == [36, 36, 36, 36]
    assert supervision.render_offset_labels.tolist() == [36, 36, 36, 36]
    assert supervision.support_view_counts.tolist() == [2, 2, 2, 2]
    assert np.all(supervision.confidence_targets > 0.9)
    assert np.all(supervision.uncertainty_px < 0.1)


def test_3dgs_multiview_supervision_accepts_geometry_subcell_seeds_without_label_collapse() -> None:
    render_seed_xy = _render_subcell_seed_xy(
        image_width=16,
        image_height=16,
        grid_width=2,
        grid_height=2,
        seed=11,
    )
    supervision = build_matcha_3dgs_multiview_coarse_supervision(
        query_view=_view(),
        render_view=_view(),
        support_views=(_view(), _view()),
        query_grid_hw=(2, 2),
        render_grid_hw=(2, 2),
        render_seed_xy=render_seed_xy,
        config=MatchaMultiviewSupervisionConfig(min_support_views=2, support_depth_tolerance_m=0.05),
    )

    assert supervision.source == "geometry_3dgs_multiview"
    assert supervision.count == 4
    assert np.allclose(supervision.query_xy, supervision.render_xy)
    assert _label_entropy_bits(supervision.render_offset_labels) > 0.0
    assert _label_entropy_bits(supervision.query_offset_labels) > 0.0
    assert not np.all(supervision.render_offset_labels == 36)


def test_subcell_fine_labels_do_not_move_coarse_supervision_pairs() -> None:
    center = build_matcha_3dgs_multiview_coarse_supervision(
        query_view=_view(),
        render_view=_view(),
        support_views=(_view(), _view()),
        query_grid_hw=(2, 2),
        render_grid_hw=(2, 2),
        config=MatchaMultiviewSupervisionConfig(min_support_views=2, support_depth_tolerance_m=0.05),
    )
    subcell = build_matcha_3dgs_multiview_coarse_supervision(
        query_view=_view(),
        render_view=_view(),
        support_views=(_view(), _view()),
        query_grid_hw=(2, 2),
        render_grid_hw=(2, 2),
        render_seed_xy=_render_subcell_seed_xy(
            image_width=16,
            image_height=16,
            grid_width=2,
            grid_height=2,
            seed=23,
        ),
        config=MatchaMultiviewSupervisionConfig(min_support_views=2, support_depth_tolerance_m=0.05),
    )

    merged, transferred = merge_fine_labels_by_cell_pair(center, subcell)

    assert transferred == center.count
    assert merged.query_indices.tolist() == center.query_indices.tolist()
    assert merged.render_indices.tolist() == center.render_indices.tolist()
    assert np.allclose(merged.query_xy, center.query_xy)
    assert np.allclose(merged.render_xy, center.render_xy)
    assert _label_entropy_bits(merged.query_offset_labels) > 0.0
    assert _label_entropy_bits(merged.render_offset_labels) > 0.0
    assert not np.all(merged.render_offset_labels == center.render_offset_labels)


def test_render_fine_label_can_transfer_when_query_subcell_crosses_cell_pair() -> None:
    coarse = MatchaCoarseSupervision(
        query_indices=np.asarray([0], dtype=np.int64),
        render_indices=np.asarray([3], dtype=np.int64),
        query_xy=np.asarray([[4.0, 4.0]], dtype=np.float64),
        render_xy=np.asarray([[12.0, 12.0]], dtype=np.float64),
        query_offset_labels=np.asarray([36], dtype=np.int64),
        render_offset_labels=np.asarray([36], dtype=np.int64),
        roundtrip_errors_px=np.asarray([0.0], dtype=np.float32),
    )
    fine = MatchaCoarseSupervision(
        query_indices=np.asarray([1], dtype=np.int64),
        render_indices=np.asarray([3], dtype=np.int64),
        query_xy=np.asarray([[12.0, 4.0]], dtype=np.float64),
        render_xy=np.asarray([[10.0, 14.0]], dtype=np.float64),
        query_offset_labels=np.asarray([7], dtype=np.int64),
        render_offset_labels=np.asarray([58], dtype=np.int64),
        roundtrip_errors_px=np.asarray([0.0], dtype=np.float32),
    )

    merged, transferred = merge_fine_labels_by_cell_pair(coarse, fine)

    assert transferred == 1
    assert merged.query_indices.tolist() == [0]
    assert merged.render_indices.tolist() == [3]
    assert merged.query_offset_labels.tolist() == [36]
    assert merged.render_offset_labels.tolist() == [58]


def test_3dgs_multiview_supervision_requires_requested_support_count() -> None:
    supervision = build_matcha_3dgs_multiview_coarse_supervision(
        query_view=_view(),
        render_view=_view(),
        support_views=(_view(x_offset=0.1), _view(x_offset=-0.1, alpha_value=0.0)),
        query_grid_hw=(2, 2),
        render_grid_hw=(2, 2),
        config=MatchaMultiviewSupervisionConfig(
            min_support_views=2,
            support_depth_tolerance_m=0.05,
            coarse_config={"alpha_threshold": 0.5},
        ),
    )

    assert supervision.source == "geometry_3dgs_multiview"
    assert supervision.count == 0


def test_3dgs_multiview_supervision_rejects_depth_inconsistent_support() -> None:
    supervision = build_matcha_3dgs_multiview_coarse_supervision(
        query_view=_view(),
        render_view=_view(),
        support_views=(_view(x_offset=0.1, depth_value=2.0),),
        query_grid_hw=(2, 2),
        render_grid_hw=(2, 2),
        config=MatchaMultiviewSupervisionConfig(min_support_views=1, support_depth_tolerance_m=0.05),
    )

    assert supervision.source == "geometry_3dgs_multiview"
    assert supervision.count == 0


def test_3dgs_multiview_builder_api_has_no_matcher_or_descriptor_inputs() -> None:
    names = set(inspect.signature(build_matcha_3dgs_multiview_coarse_supervision).parameters)
    forbidden_fragments = ("match", "descriptor", "feature", "radio")

    assert not any(fragment in name.lower() for name in names for fragment in forbidden_fragments)
