import numpy as np
import pytest
from scripts.diagnose_av2_density_scan_ratio import profile, trace, summarize

PARAMS = dict(n_rings=2, n_sectors=4, max_range=20., scan_ratio_threshold=.2,
              min_map_height=.5, ground_margin=.2, min_votes=None,
              votes_fraction=.5, votes_floor=3)


def test_profile_ignores_outside_points_and_handles_singletons():
    pts = np.array([[5., 0, 0], [5., 0, 2], [-5., 0, 4], [25., 0, 100]])
    bins, valid, height, count = profile(pts, np.zeros(3), PARAMS)
    assert valid.tolist() == [True, True, True, False]
    assert height[bins[0]] == 2 and count[bins[0]] == 2
    assert height[bins[2]] == 0 and count[bins[2]] == 1
    assert count.sum() == 3


def test_replay_preserves_unobserved_low_and_height_ratio_stages():
    points = np.array([[5., 0, 0], [5., 0, 2], [-5., 0, 0], [0., 15, 0], [30., 0, 0]])
    query = np.array([[5., 0, 0], [5., 0, 2], [-5., 0, 0]])
    r = trace(points, [(query, np.zeros(3))]*3, PARAMS)
    assert r['states'][0].tolist() == [3, 3, 2, 1, 0]
    assert not r['removed'].any()
    assert r['observed'].tolist() == [3, 3, 3, 0, 0]


def evidence(states, threshold):
    s = np.array(states, np.uint8)
    votes = (s == 5).sum(axis=0)
    t = np.array(threshold)
    return dict(states=s, threshold=t, votes=votes, removed=votes >= t)


def test_lost_votes_partition_destinations_despite_lower_threshold():
    dense = evidence([[5,5], [5,5], [5,5], [5,5]], [3,3])
    sparse = evidence([[3,5], [4,5], [1,5], [5,5]], [2,3])
    r = summarize(dense, sparse, np.arange(2), np.array([True, False]))
    lost = r['moving_lost_removal']
    assert lost['points'] == 1 and lost['lost_votes'] == 3
    assert lost['threshold_lowered_points'] == 1
    assert lost['lost_vote_destination']['height_ratio_too_large'] == 1
    assert lost['lost_vote_destination']['ground_reverted'] == 1
    assert lost['lost_vote_destination']['empty_query_column'] == 1
    assert sum(map(sum,lost['stage_transition_dense_to_sparse'])) == 4
    assert r['static_added_removal']['points'] == 0


def test_subset_cannot_increase_threshold_or_observation():
    dense = evidence([[1]], [1])
    with pytest.raises(ValueError, match='observed'):
        summarize(dense, evidence([[3]], [1]), np.array([0]), np.array([True]))
    with pytest.raises(ValueError, match='threshold'):
        summarize(dense, evidence([[1]], [2]), np.array([0]), np.array([True]))


def test_flat_query_votes_for_elevated_map_body_and_reverts_ground():
    ground = np.array([[x,y,0.] for x in [5.,5.5,6.] for y in [.2,.5,.8]])
    points = np.vstack([ground, [5.5,.5,2.]])
    result = trace(points, [(ground,np.zeros(3))]*3, PARAMS)
    np.testing.assert_array_equal(result['states'][:,:-1], 4)
    np.testing.assert_array_equal(result['states'][:,-1], 5)
    assert result['removed'].tolist() == [False]*9+[True]
