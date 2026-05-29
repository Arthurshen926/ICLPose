from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.visualize_stage_c1_selection_qualitative import (
    compute_similarity_margin_map,
    group_energy_from_transform,
    make_contact_sheet,
)


def test_margin_map_reports_top1_minus_top2_per_token() -> None:
    query_feature_map = np.asarray(
        [
            [[1.0, 0.0]],
            [[0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    landmark_features = np.asarray(
        [
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    margin_map = compute_similarity_margin_map(query_feature_map, landmark_features, block_size=2)

    expected = 1.0 - np.sqrt(0.5)
    assert margin_map.shape == (1, 2)
    np.testing.assert_allclose(margin_map, [[expected, expected]], atol=1e-6)


def test_group_energy_sums_projection_rows_by_group() -> None:
    transform = np.asarray(
        [
            [1.0, 2.0],
            [0.0, 3.0],
            [4.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )

    energy = group_energy_from_transform(transform, group_size=2)

    np.testing.assert_allclose(energy, [14.0, 18.0], atol=1e-6)


def test_contact_sheet_writes_titled_grid(tmp_path: Path) -> None:
    panels = [
        np.full((8, 10, 3), 50, dtype=np.uint8),
        np.full((8, 10, 3), 100, dtype=np.uint8),
        np.full((8, 10, 3), 150, dtype=np.uint8),
    ]

    sheet = make_contact_sheet(panels, labels=["a", "b", "c"], columns=2, cell_width=24, cell_height=16)

    assert sheet.shape == (32, 48, 3)
    assert sheet.dtype == np.uint8
    assert int(sheet.mean()) > 0
