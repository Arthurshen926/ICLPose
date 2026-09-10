import numpy as np
from feature_extract.tools.vfm.audit_goal_maplet_mapping_crossplane_retrieval import physical_labels


def test_close_cross_plane_and_cell_boundary_are_not_forced_negatives():
    np.testing.assert_array_equal(physical_labels([.1,.1,.3,.6],np.array([True,False,True,False])),[1,-1,-1,0])
def test_conflicting_mapping_targets_are_not_selected_by_distance():
    import numpy as np
    from feature_extract.tools.vfm.audit_goal_maplet_mapping_crossplane_retrieval import ambiguous_source_tokens
    bad=ambiguous_source_tokens(np.array([0,0,0,1]),np.array([2,2,3,2]),np.array([[0.,0,0],[1.,0,0],[0.,0,0],[2.,0,0]]))
    assert bad=={(0,2)}
