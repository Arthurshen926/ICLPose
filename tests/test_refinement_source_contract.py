import numpy as np
import pytest
from feature_extract.tools.vfm.refinement_source_contract import validate_crossroute_support


def check(rows=(1,2), members=(1,), names=('seq7__a',), excluded='seq7'):
    return validate_crossroute_support(names, np.array(rows, dtype=int), np.array(members, dtype=int),
                                      np.array([0,1,2]), ['seq7__a','seq1__b','seq2__c'], excluded)


def test_source_exclusion_allows_distinct_route_geometry_outside_region():
    assert check()['shared_scene_geometry']


@pytest.mark.parametrize('kwargs', [dict(rows=(0,)), dict(members=(0,)),
                                   dict(names=('seq1__a',)), dict(rows=(-1,)), dict(rows=(3,))])
def test_source_exclusion_rejects_contamination_and_bad_identity(kwargs):
    with pytest.raises(ValueError):
        check(**kwargs)
