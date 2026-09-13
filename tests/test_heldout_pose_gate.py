import pytest
from feature_extract.tools.vfm.heldout_pose_gate import heldout_gate


def test_absence_is_not_contradiction():
    assert heldout_gate((0,-100),(0,-100),'veto')
    assert not heldout_gate((0,-100),(0,-100),'corroborate')


def test_supported_baseline_can_veto_a_worse_candidate():
    assert not heldout_gate((8,-50),(4,-40),'veto')
    assert heldout_gate((5,-50),(4,-40),'veto')


def test_corroboration_needs_support_and_strict_score_improvement():
    assert heldout_gate((8,-50),(8,-40),'corroborate')
    assert not heldout_gate((8,-50),(8,-50),'corroborate')
    assert not heldout_gate((0,-100),(5,-40),'corroborate')


def test_unknown_gate_fails_explicitly():
    with pytest.raises(ValueError):heldout_gate((0,0),(0,0),'unknown')
