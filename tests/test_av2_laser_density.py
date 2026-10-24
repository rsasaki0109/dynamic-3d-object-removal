import numpy as np
import pytest

from scripts.experiment_av2_laser_density import METHODS, VARIANTS, select_rows, subset, score_case


@pytest.mark.parametrize('modulus', [2, 4])
def test_complementary_phases_cover_each_source_row_once(modulus):
    ids = np.tile(np.arange(63, -1, -1), 2)
    masks = [select_rows(ids, v) for v in VARIANTS if v['modulus'] == modulus]
    np.testing.assert_array_equal(np.sum(masks, axis=0), np.ones(len(ids)))
    for mask in masks:
        assert mask.sum() == len(ids) // modulus


@pytest.mark.parametrize('ids', [[-1], [64], [1.5], [[1, 2]]])
def test_reject_invalid_laser_metadata(ids):
    with pytest.raises(ValueError):
        select_rows(ids, VARIANTS[0])


def test_subset_preserves_source_order_labels_and_origins():
    scans = [(np.arange(12).reshape(4, 3), np.array([1., 2., 3.])),
             (np.arange(9).reshape(3, 3), np.array([4., 5., 6.]))]
    gt = np.array([True, False, False, True, False, True, True])
    chosen, labels, indices = subset(scans, gt, [np.array([3, 0, 2, 1]), np.array([0, 1, 2])], VARIANTS[1])
    np.testing.assert_array_equal(indices, [1, 2, 4, 6])
    np.testing.assert_array_equal(labels, gt[indices])
    np.testing.assert_array_equal(chosen[0][0], scans[0][0][[1, 2]])
    np.testing.assert_array_equal(chosen[1][1], scans[1][1])


def test_empty_phase_cannot_silently_drop_frame():
    with pytest.raises(ValueError, match='empty frame'):
        subset([(np.zeros((2, 3)), np.zeros(3))], np.zeros(2, dtype=bool), [np.array([1, 3])], VARIANTS[1])


def test_matched_population_has_zero_delta_despite_selection_bias():
    gt = np.array([True, False])
    indices = np.array([1, 3])
    dense = {m: np.array([False, True, True, False]) for m in METHODS}
    actual = {m: p[indices] for m, p in dense.items()}
    result = score_case(actual, dense, indices, gt)
    for m in METHODS:
        assert all(v == 0 for v in result[m]['metric_delta'].values())
        assert all(v == 0 for v in result[m]['decision_changes'].values())


def test_decision_changes_account_for_both_label_classes():
    gt = np.array([True, True, False, False])
    dense = {m: np.array([False, True, False, True]) for m in METHODS}
    actual = {m: ~p for m, p in dense.items()}
    result = score_case(actual, dense, np.arange(4), gt)
    for m in METHODS:
        assert set(result[m]['decision_changes'].values()) == {1}
