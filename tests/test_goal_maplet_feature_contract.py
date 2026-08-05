import numpy as np
import pytest

from feature_extract.vfm.localization_goal_maplet.feature_contract import FieldFeatureContract


def test_feature_contract_rejects_implicit_or_unknown_readout():
    with pytest.raises(ValueError):
        FieldFeatureContract("field", "dimension_guess", "", "render")
    with pytest.raises(ValueError):
        FieldFeatureContract("field", "surface_maplet_mapper", "", "render")
