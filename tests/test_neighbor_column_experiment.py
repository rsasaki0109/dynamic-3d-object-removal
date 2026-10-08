import numpy as np

import dynamic_object_removal as core
from scripts.experiment_neighbor_columns import fill_neighbors, scan_votes


def test_neighbor_pooling_wraps_sectors_without_crossing_rings_or_overwriting_data():
    high = np.array([1., -np.inf, 3., -np.inf, 9., -np.inf, -np.inf, -np.inf])
    low = np.array([0., np.inf, 1., np.inf, 8., np.inf, np.inf, np.inf])
    counts = np.array([2, 0, 2, 0, 2, 0, 0, 0])
    h, l, c, borrowed = fill_neighbors(high, low, counts, 2, 4, "both")
    assert borrowed.tolist() == [False, True, False, True, False, False, False, False]
    assert h[1] == h[3] == 3
    assert l[1] == l[3] == 0
    assert c[1] == c[3] == 4
    assert h[0] == 1 and h[4] == 9
    assert counts.tolist() == [2, 0, 2, 0, 2, 0, 0, 0]
    _, _, _, one_side = fill_neighbors(high, low, counts, 2, 4, "either")
    assert one_side[5] and one_side[7] and not one_side[6]


def test_no_neighbor_mode_matches_native_per_scan_masks():
    rng = np.random.default_rng(42)
    points = rng.uniform([-15, -15, -1], [15, 15, 3], (3000, 3))
    query = points[::8].copy()
    origin = np.array([1., -2., .5])
    params = {"n_rings": 5, "n_sectors": 24, "max_range": 30.,
              "scan_ratio_threshold": .8, "min_map_height": .5, "ground_margin": .2}
    expected_dynamic, expected_observed = core._scan_ratio_dynamic(points, query, origin, **params)
    dynamic, observed, inferred = scan_votes(points, query, origin, params, "none")
    assert expected_dynamic.any() and expected_observed.any()
    np.testing.assert_array_equal(dynamic, expected_dynamic)
    np.testing.assert_array_equal(observed, expected_observed)
    assert not inferred.any()


def test_lower_z_continuity_rejects_height_steps_and_missing_extrema():
    from scripts.experiment_neighbor_columns import ground_aligned_support
    borrowed = np.array([True, True, True, True, False])
    query_low = np.array([0., .5, np.inf, 0., 0.])
    map_low = np.array([.125, 0., 0., np.inf, 0.])
    counts = np.array([2, 2, 2, 0, 2])
    assert ground_aligned_support(borrowed, query_low, map_low, counts, .125).tolist() == [True, False, False, False, False]


def test_lower_z_gate_preserves_native_votes_and_only_reduces_inference():
    rng = np.random.default_rng(64)
    points = rng.uniform([-15, -15, -1], [15, 15, 3], (3000, 3))
    query = points[::20].copy()
    origin = np.array([1., -2., .5])
    params = {"n_rings": 10, "n_sectors": 80, "max_range": 30.,
              "scan_ratio_threshold": .8, "min_map_height": .5, "ground_margin": .2}
    native, obs, _ = scan_votes(points, query, origin, params, "none")
    either, eo, ei = scan_votes(points, query, origin, params, "either")
    gated, go, gi = scan_votes(points, query, origin, params, "ground_aligned")
    assert ei.any() and (ei & ~gi).any()
    np.testing.assert_array_equal(gated[obs], native[obs])
    np.testing.assert_array_equal(go[obs], obs[obs])
    assert not (gi & ~ei).any() and not (go & ~eo).any()
    assert not (gated & ~either).any()
