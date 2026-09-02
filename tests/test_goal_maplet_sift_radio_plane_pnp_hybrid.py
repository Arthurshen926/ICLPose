from feature_extract.tools.vfm.select_goal_maplet_sift_radio_plane_pnp_hybrid import (
    _use_sift,
)


def test_sift_hybrid_support_gate_is_inclusive_at_sixteen() -> None:
    assert not _use_sift(0, 100)
    assert not _use_sift(3, 15)
    assert _use_sift(1, 16)
