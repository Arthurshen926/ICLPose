from pathlib import Path
from feature_extract.tools.vfm.replay_goal_maplet_corrected_baseline import commands


def test_replay_explicitly_preserves_historical_moge_and_selection_contract():
    jobs=dict(commands(*[Path(s) for s in ['corr','atlas','map','planes','contributors','out']],
        {'query_support_weighting':'sqrt_visible_fraction','plane_association_policy':'many_query_fragments_per_map_plane'},
        {'fixed_hypothesis_selection_policy':'calibrated_gaussian_null'}))
    assert jobs['moge'][jobs['moge'].index('--query_support_weighting')+1]=='sqrt_visible_fraction'
    assert jobs['moge'][jobs['moge'].index('--plane_association_policy')+1]=='many_query_fragments_per_map_plane'
    assert jobs['final'][jobs['final'].index('--hypothesis_selection_policy')+1]=='calibrated_gaussian_null'
    assert jobs['pnp'][jobs['pnp'].index('--solver_policy')+1]=='unique_token_lm'
    assert jobs['view'][jobs['view'].index('--solver_policy')+1]=='unique_token_lm'


def test_legacy_missing_selection_field_means_nearest_not_new_default():
    jobs=dict(commands(*[Path(s) for s in ['corr','atlas','map','planes','contributors','out']],
        {'query_support_weighting':'uniform','plane_association_policy':'one_to_one'},{}))
    assert jobs['final'][jobs['final'].index('--hypothesis_selection_policy')+1]=='nearest_reprojection'
