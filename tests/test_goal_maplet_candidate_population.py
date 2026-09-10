from feature_extract.tools.vfm.audit_goal_maplet_candidate_population import population


def test_overlapping_plane_observations_are_one_image_token():
    r=population([0,0,0,1],[2,2,3,2],[1,-1,0,-1])
    assert r=={'independent_tokens':3,'with_positive_evidence':1,
               'without_positive_but_with_ambiguous_evidence':1,'only_negative_evidence':1}
