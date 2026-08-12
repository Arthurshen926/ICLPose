import numpy as np
import pytest

from feature_extract.tools.vfm.build_goal_maplet_physical_map import _parse_args
from feature_extract.tools.vfm.build_v6_contributor_cache import (
    _declared_clean_source_indices,
)
from feature_extract.vfm.vfm_2dgs_mapping import SurfaceElementMap


def _surface(path):
    SurfaceElementMap(
        element_ids=np.asarray([10, 11, 12], dtype=np.int64),
        parent_gaussian_indices=np.asarray([4, 2, 4], dtype=np.int64),
        centers=np.zeros((3, 3), dtype=np.float64),
        tangent1=np.tile([1.0, 0.0, 0.0], (3, 1)),
        tangent2=np.tile([0.0, 1.0, 0.0], (3, 1)),
        normals=np.tile([0.0, 0.0, 1.0], (3, 1)),
        scale1=np.ones(3, dtype=np.float32),
        scale2=np.ones(3, dtype=np.float32),
        opacity=np.ones(3, dtype=np.float32),
        area=np.ones(3, dtype=np.float32),
        adjacency=(np.asarray([], dtype=np.int64),) * 3,
    ).save_npz(path)


def test_contributor_clean_surface_elements_resolve_parent_source_ids(tmp_path):
    path = tmp_path / "surface.npz"
    _surface(path)
    indices, policy, declaration = _declared_clean_source_indices(
        6, clean_surface_elements=str(path)
    )
    np.testing.assert_array_equal(indices, np.asarray([2, 4]))
    assert policy == "declared_surface_element_parent_source_mask"
    assert declaration == path


def test_contributor_clean_geometry_declarations_are_exclusive(tmp_path):
    with pytest.raises(ValueError, match="mutually exclusive"):
        _declared_clean_source_indices(
            2,
            clean_gaussian_ply=str(tmp_path / "clean.ply"),
            clean_surface_elements=str(tmp_path / "surface.npz"),
        )


def test_physical_map_cli_accepts_strict_surface_declaration():
    args = _parse_args([
        "--surface_elements", "surface.npz",
        "--clean_surface_elements",
        "--legacy_maplets", "maplets.npz",
        "--region_map", "region.npz",
        "--mapping_pose_file", "poses.txt",
        "--output_map", "physical.npz",
        "--audit_json", "audit.json",
    ])
    assert args.clean_surface_elements is True
    assert args.clean_gaussian_ply is None
