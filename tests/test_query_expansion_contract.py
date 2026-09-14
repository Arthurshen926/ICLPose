import numpy as np
import pytest
from feature_extract.tools.vfm.run_overlap_lod_frontend import expand_query_tokens


def test_expansion_preserves_old_support_and_excludes_heldout_and_invalid_new_tokens():
    # Old support survives a later validity change; new invalid observations do not enter.
    result = expand_query_tokens([4,1], [True,False,True,True,False,True], [0,5])
    assert result.tolist() == [4,1,2,3]


@pytest.mark.parametrize('original,held', [([1],[1]), ([1,1],[]), ([-1],[]), ([6],[]), ([1],[-1])])
def test_expansion_rejects_contamination_or_bad_identity(original,held):
    with pytest.raises(ValueError):
        expand_query_tokens(original, np.ones(6,bool), held)
