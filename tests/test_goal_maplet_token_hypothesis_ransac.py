import cv2
import numpy as np
from feature_extract.tools.vfm.token_hypothesis_ransac import solve,score_pose,canonical_hypotheses


def fixture():
    rng=np.random.default_rng(4);good=rng.uniform([-1,-1,4],[1,1,8],(20,3))
    bad=rng.uniform([-3,-3,3],[3,3,9],(20,3));world=np.r_[good,bad]
    K=np.array([[140.,0.,128.],[0.,145.,72.],[0.,0.,1.]])
    pixels=cv2.projectPoints(good,np.zeros(3),np.zeros(3),K,np.array([.02,0,0,0,0.]))[0].reshape(-1,2)
    return world,np.tile(np.arange(20),2),np.tile(pixels,(2,1)),K


def test_recovers_pose_with_mutually_exclusive_hypotheses():
    world,tokens,pixels,K=fixture()
    pose=solve(world,tokens,K,.02,np.arange(len(world)),pixels)
    assert pose is not None
    np.testing.assert_allclose(pose,np.eye(4),atol=1e-6)


def test_duplicate_and_permutation_invariance():
    world,tokens,pixels,K=fixture()
    first=solve(world,tokens,K,.02,np.arange(len(world)),pixels)
    rows=np.r_[np.arange(len(world))[::-1],np.arange(0,len(world),2)]
    again=solve(world[rows],tokens[rows],K,.02,np.arange(len(rows)),pixels[rows])
    np.testing.assert_array_equal(first,again)


def test_score_radial_projection_matches_opencv_and_counts_tokens_once():
    world,tokens,pixels,K=fixture()
    groups=[np.flatnonzero(tokens==i) for i in np.unique(tokens)]
    key,selected=score_pose(np.eye(4),world,pixels,groups,K,.02)
    assert key[0]==20 and len(selected)==20 and abs(key[1])<1e-20
    pose=np.eye(4);pose[2,3]=-20
    assert score_pose(pose,world,pixels,groups,K,.02)[0][0]==0


def test_vectorized_score_is_exact_reference_score():
    world,tokens,pixels,K=fixture()
    world,tokens,pixels=canonical_hypotheses(world,tokens,pixels)
    groups=[np.flatnonzero(tokens==i) for i in np.unique(tokens)]
    starts=np.flatnonzero(np.r_[True,tokens[1:]!=tokens[:-1]])
    pose=np.eye(4);pose[:3,3]=[.1,.2,.3]
    reference=score_pose(pose,world,pixels,groups,K,.02)[0]
    fast=score_pose(pose,world,pixels,groups,K,.02,group_starts=starts,return_selected=False)[0]
    assert reference==fast


def test_collinear_geometry_is_rejected():
    world=np.c_[np.arange(8),np.zeros(8),np.ones(8)*5].astype(float)
    assert solve(world,np.arange(8),np.eye(3),0,np.arange(8),np.ones((8,2)),iterations=8) is None


def test_lm_cannot_replace_pose_with_worse_unique_token_score(monkeypatch):
    world,tokens,pixels,K=fixture()
    monkeypatch.setattr(cv2,'solvePnPRefineLM',lambda *a,**kw:(np.zeros(3),np.array([100.,0.,0.])))
    pose=solve(world,tokens,K,.02,np.arange(len(world)),pixels)
    np.testing.assert_allclose(pose,np.eye(4),atol=1e-5)


def test_guidance_is_duplicate_permutation_invariant_and_budget_matched():
    world,tokens,pixels,K=fixture()
    planes=tokens%3;scores=np.linspace(0,1,len(tokens));rows=np.arange(len(tokens));stats={}
    first=solve(world,tokens,K,.02,rows,pixels,iterations=512,hypothesis_budget=64,
                sampling_policy='geometry_score',planes=planes,scores=scores,stats=stats)
    assert stats['scored_hypotheses']==64 and stats['budget_reached']
    duplicate=np.r_[rows[::-1],rows[::2]]
    second=solve(world[duplicate],tokens[duplicate],K,.02,np.arange(len(duplicate)),pixels[duplicate],
                 iterations=512,hypothesis_budget=64,sampling_policy='geometry_score',
                 planes=planes[duplicate],scores=scores[duplicate])
    np.testing.assert_array_equal(first,second)


def test_guided_samples_use_four_independent_tokens():
    from feature_extract.tools.vfm.token_hypothesis_ransac import guided_sample
    groups=[np.array([i,i+8]) for i in range(8)]
    rng=np.random.default_rng(3)
    for _ in range(100):
        sample=guided_sample(rng,groups,np.c_[np.arange(8),np.zeros(8)],np.arange(16)%3,np.zeros(16),'geometry_score')
        assert len(np.unique(sample%8))==4


def test_opencv_lm_guard_retains_consensus_when_lm_proposal_is_bad(monkeypatch):
    from feature_extract.tools.vfm.evaluate_goal_maplet_direct_plane_pnp_multihypothesis import _solve
    world,tokens,pixels,K=fixture()
    monkeypatch.setattr(cv2,'solvePnPRefineLM',lambda *a,**kw:(np.zeros(3),np.array([100.,0.,0.])))
    stats={}
    pose=_solve(world,tokens,K,.02,np.arange(20),pixels,guard_lm=True,sampling_stats=stats)
    np.testing.assert_allclose(pose,np.eye(4),atol=1e-5)
    assert not stats['lm_accepted']


def test_equal_scores_are_exact_geometry_control_not_random_stream_change():
    from feature_extract.tools.vfm.token_hypothesis_ransac import guided_sample
    groups=[np.array([i,i+8]) for i in range(8)];centers=np.c_[np.arange(8),np.zeros(8)]
    planes=np.arange(16)%3;scores=np.ones(16)
    first=np.random.default_rng(5);second=np.random.default_rng(5)
    for _ in range(50):
        np.testing.assert_array_equal(guided_sample(first,groups,centers,planes,scores,'geometry'),
                                      guided_sample(second,groups,centers,planes,scores,'geometry_score'))


def test_row_and_token_controls_match_scored_model_budget():
    world,tokens,pixels,K=fixture()
    for policy in ['uniform','row_uniform']:
        stats={}
        pose=solve(world,tokens,K,.02,np.arange(len(world)),pixels,iterations=1024,
                   hypothesis_budget=128,sampling_policy=policy,stats=stats)
        assert pose is not None and np.isfinite(pose).all()
        assert stats['scored_hypotheses']==128 and stats['budget_reached']
        np.testing.assert_allclose(pose,np.eye(4),atol=1e-5)
