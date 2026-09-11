import numpy as np
from feature_extract.tools.vfm.rerank_goal_maplet_memory_pose_pool import soft_token_support


def test_precision_support_counts_unique_token_and_rejects_behind_camera():
    pose=np.eye(4);K=np.eye(3);tokens=np.array([0,1]);world=np.array([[1.5,1.5,1.],[5.5,1.5,1.]])
    a=soft_token_support(pose,world,tokens,K,0,np.array([.5,.8]))
    ids=np.array([0,0,1]);b=soft_token_support(pose,world[ids],tokens[ids],K,0,np.array([.5,.5,.8]))
    assert np.isclose(a,1.3) and np.isclose(a,b)
    assert soft_token_support(pose,-world,tokens,K,0,np.ones(2))==0


def test_precision_support_prefers_sharper_equal_count_alignment():
    pose=np.eye(4);K=np.eye(3);tokens=np.array([0,1]);world=np.array([[1.5,1.5,1.],[5.5,1.5,1.]])
    moved=pose.copy();moved[0,3]=2.
    assert soft_token_support(pose,world,tokens,K,0,np.ones(2))>soft_token_support(moved,world,tokens,K,0,np.ones(2))


def test_zero_correspondence_query_is_not_removed_from_evaluation():
    import pytest
    from feature_extract.tools.vfm.evaluate_goal_maplet_structured_memory_pose import evaluation_images
    names=np.array(['seq9__a','seq9__b'])
    np.testing.assert_array_equal(evaluation_images(np.array([0,0]),names,list(names)),[0,1])
    with pytest.raises(ValueError,match='missing'):evaluation_images(np.array([0]),names,['unlisted'])
