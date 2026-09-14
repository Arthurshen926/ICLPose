from feature_extract.tools.vfm.pose_evidence_selection import select_candidate


def test_failed_top_candidate_does_not_hide_confirmed_runner_up():
    choice,reason=select_candidate([.5,.99,.8],[False,True,True],[(10,-1)]*3,[(10,-5),(9,-1),(12,-2)],.6)
    assert (choice,reason)==(2,'relative_and_bank')


def test_rescue_requires_improvement_in_both_token_domains():
    args=([.5,.2],[False,True],[(10,-5),(12,-2)],[(10,-5),(12,-2)],.6)
    assert select_candidate(*args)==(0,'retain_reference')
    assert select_candidate(*args,policy='dual_support')==(1,'dual_support_rescue')
    assert select_candidate(*args[:3],[(10,-5),(9,-2)],.6,policy='dual_support')[0]==0


def test_local_or_tied_candidates_cannot_trigger_rescue():
    assert select_candidate([.5,.9],[False,False],[(10,-5),(12,-2)],[(10,-5),(12,-2)],.6,'dual_support')[0]==0
    assert select_candidate([.5,.9],[False,True],[(10,-5),(12,-2)],[(10,-5),(10,-5)],.6,'dual_support')[0]==0
