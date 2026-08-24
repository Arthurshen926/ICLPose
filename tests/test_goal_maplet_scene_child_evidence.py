import numpy as np
import pytest

from feature_extract.vfm.localization_goal_maplet.fine_support_selection import (
    EVIDENCE_BLOCK_CAPPED_SUM,
    EVIDENCE_TOKEN_SUM,
)
from feature_extract.vfm.localization_goal_maplet.pure_retrieval import (
    all_radio_token_coordinates,
)
from feature_extract.vfm.localization_goal_maplet.scene_child_evidence import (
    EVIDENCE_LEGACY_MACRO_TOP4,
    aggregate_scene_child_evidence,
)
from test_goal_maplet_pure_retrieval import _physical


def _inputs():
    xy = all_radio_token_coordinates(4, 4)
    rows = np.full((16, 2), -1, dtype=np.int64)
    probabilities = np.zeros((16, 2), dtype=np.float32)
    rows[:, 0] = 0
    probabilities[:, 0] = 0.1
    rows[0, 1] = 1
    probabilities[0, 1] = 0.4
    return xy, rows, probabilities


def test_local_token_sum_preserves_more_repeated_support_than_macro_max():
    physical = _physical()
    xy, rows, probabilities = _inputs()
    macro, _ = aggregate_scene_child_evidence(
        xy, rows, probabilities, physical,
        token_height=4, token_width=4,
        semantics=EVIDENCE_LEGACY_MACRO_TOP4,
    )
    token_sum, audit = aggregate_scene_child_evidence(
        xy, rows, probabilities, physical,
        token_height=4, token_width=4,
        semantics=EVIDENCE_TOKEN_SUM,
    )
    assert token_sum[0] > macro[0]
    assert audit["parent_evidence_remultiplied"] is False


def test_scene_parent_mask_removes_only_children_outside_selected_parent():
    physical = _physical()
    xy, rows, probabilities = _inputs()
    score, audit = aggregate_scene_child_evidence(
        xy, rows, probabilities, physical,
        token_height=4, token_width=4,
        semantics=EVIDENCE_BLOCK_CAPPED_SUM,
        scene_parent_ids=physical.maplet_ids[:1],
    )
    allowed = physical.child_parent_rows == 0
    assert np.all(score[~allowed] == 0.0)
    assert score[0] > 0.0
    assert audit["scene_parent_mask_applied"] is True
    assert audit["scene_parent_count"] == 1


def test_scene_parent_mask_fails_closed_on_unknown_or_duplicate_ids():
    physical = _physical()
    xy, rows, probabilities = _inputs()
    with pytest.raises(ValueError):
        aggregate_scene_child_evidence(
            xy, rows, probabilities, physical,
            token_height=4, token_width=4,
            semantics=EVIDENCE_TOKEN_SUM,
            scene_parent_ids=np.asarray([999999], dtype=np.int64),
        )
    with pytest.raises(ValueError):
        aggregate_scene_child_evidence(
            xy, rows, probabilities, physical,
            token_height=4, token_width=4,
            semantics=EVIDENCE_TOKEN_SUM,
            scene_parent_ids=np.repeat(physical.maplet_ids[:1], 2),
        )
