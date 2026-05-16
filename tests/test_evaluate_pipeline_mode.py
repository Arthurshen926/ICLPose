import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from pose_refine.evaluate_pipeline import _resolve_eval_mode_metadata


def test_resolve_eval_mode_marks_default_loftr_as_oracle():
    meta = _resolve_eval_mode_metadata(None, "loftr")

    assert meta["eval_mode"] == "oracle"
    assert meta["uses_gt_query_center"] is True
    assert meta["deployable"] is False


def test_resolve_eval_mode_rejects_deploy_with_oracle_loftr():
    with pytest.raises(ValueError, match="oracle"):
        _resolve_eval_mode_metadata("deploy", "loftr")


def test_resolve_eval_mode_allows_regression_deploy():
    meta = _resolve_eval_mode_metadata("deploy", "regression")

    assert meta["eval_mode"] == "deploy"
    assert meta["uses_gt_query_center"] is False
    assert meta["deployable"] is True
