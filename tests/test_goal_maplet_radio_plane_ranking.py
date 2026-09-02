import pytest

from feature_extract.tools.vfm.build_goal_maplet_radio_plane_ranking import (
    _validate_plane_inventory,
)


class _PlanarRows:
    def __init__(self, count: int) -> None:
        self.plane_ids = list(range(count))


def test_plane_ranking_rejects_mismatched_map_inventory() -> None:
    with pytest.raises(ValueError, match="row mismatch"):
        _validate_plane_inventory(_PlanarRows(1998), {"plane_count": 2010}, 2010)


def test_plane_ranking_accepts_exact_map_inventory() -> None:
    _validate_plane_inventory(_PlanarRows(2010), {"plane_count": 2010}, 2010)
