from feature_extract.tools.vfm.strong_local_selection import choose_local,calibrate_forward,threshold_harm


def row(name='a',candidate=(.08,.2)):
    return dict(name=name,valid_local_pair=True,support=6,base_error=(.09,.2),candidate_error=candidate)


def test_explicit_abstention_rejects_even_saturated_probability():
    assert choose_local([row()],[1.],1.,enabled=False)==-1


def test_forward_calibration_cannot_count_repeated_candidate_pairs_as_queries():
    assert calibrate_forward([[row(),row()] for _ in range(6)],[[.9,.8]]*6)==(1.,False)


def test_calibration_accounts_for_all_threshold_harms():
    assert threshold_harm((.24,1.5),(.26,1.5))
    groups=[[row(str(i))] for i in range(5)]
    assert calibrate_forward(groups,[[.8]]*5)==(.5,True)
    groups[0][0]['candidate_error']=(.11,.2)
    assert calibrate_forward(groups,[[.8]]*5)==(1.,False)


def test_multiscale_agreement_rejects_fine_only_gain_and_checks_runner_up():
    from feature_extract.tools.vfm.strong_local_selection import choose_multiscale_local
    first=dict(row(),features=[-.1]+[0.]*5+[.9]+[0.]*5)
    second=dict(row(),features=[.1]+[0.]*5+[.2]+[0.]*5)
    assert choose_multiscale_local([first])==-1
    assert choose_multiscale_local([first,second])==1
    second['features'][0]=0.
    assert choose_multiscale_local([second])==-1
