import numpy as np
from scripts.analyze_neighbor_errors import changed_groups, evidence


def test_change_groups_partition_decision_changes_by_ground_truth():
    native = np.array([0, 0, 1, 1, 0, 1], bool)
    candidate = np.array([1, 1, 0, 0, 0, 1], bool)
    gt = np.array([1, 0, 1, 0, 1, 0], bool)
    groups = changed_groups(native, candidate, gt)
    np.testing.assert_array_equal(sum(groups.values()), native != candidate)
    assert [np.flatnonzero(m).tolist() for m in groups.values()] == [[0], [1], [2], [3]]


def test_inferred_observations_can_lose_removal_without_losing_votes(monkeypatch):
    from scripts import analyze_neighbor_errors as module
    def fake_votes(points, scan, origin, params, support):
        dynamic = np.array([bool(scan[0, 0])])
        observed = np.array([bool(scan[0, 0]) or support == 'either'])
        inferred = np.array([not bool(scan[0, 0]) and support == 'either'])
        return dynamic, observed, inferred
    monkeypatch.setattr(module, 'scan_votes', fake_votes)
    scans = [(np.array([[float(i < 3), 0, 0]]), np.zeros(3)) for i in range(8)]
    params = dict(votes_floor=3, votes_fraction=.5)
    native = evidence(np.zeros((1, 3)), scans, params, 'none')
    candidate = evidence(np.zeros((1, 3)), scans, params, 'either')
    assert native['votes'][0] == candidate['votes'][0] == 3
    assert native['threshold'][0] == 3 and candidate['threshold'][0] == 4
    assert candidate['inferred'][0] == 5 and candidate['inferred_votes'][0] == 0
