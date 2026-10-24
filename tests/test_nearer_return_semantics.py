import numpy as np
import pytest

from scripts.classify_nuscenes_nearer_semantics import align_labels, counts, summarize


def test_raw_row_indices_survive_ground_and_strict_close_square_exclusion():
    raw = np.array([[2, 0, -2, 0, 0], [.5, .5, 0, 0, 0], [2, 0, 0, 0, 0],
                    [1, 0, -1.6, 0, 0], [0, 1, 0, 0, 0]], dtype=np.float32)
    # Use a binary-exact boundary to avoid a float32 encoding assumption.
    raw[3, 2] = -1.5
    points, labels, row = align_labels(raw, np.array([1, 2, 17, 3, 24], np.uint8), -1.5, 1.)
    assert row.tolist() == [2, 4]
    assert labels.tolist() == [17, 24]
    np.testing.assert_array_equal(points, raw[[2, 4], :3])


def test_wrong_raw_label_count_and_unknown_semantic_id_are_rejected():
    with pytest.raises(ValueError, match="row-aligned"):
        align_labels(np.zeros((3, 5)), np.array([1, 2]), -1.6, 1)
    with pytest.raises(ValueError, match="category mapping"):
        counts(np.array([31], np.uint8), {17: "vehicle.car"})


def test_repeated_events_preserve_annotation_subgroups_and_own_scan_population():
    cats = {17: "vehicle.car", 24: "flat.driveable_surface", 28: "static.manmade"}
    result = summarize(np.array([28, 28, 24, 17], np.uint8), np.array([3, 3, 3, 0], np.uint8),
                       np.array([1, 0, 0, 1], bool), cats)
    assert result["all_queries"]["events"] == 4
    assert result["all_queries"]["outside_boxes_semantic_counts"] == {
        "vehicle.car": 0, "flat.driveable_surface": 1, "static.manmade": 2}
    assert result["same_acquisition_scan"]["events"] == 2
    assert result["same_acquisition_scan"]["outside_boxes_semantic_counts"]["static.manmade"] == 1
    assert sum(result["other_query_scans"]["semantic_counts"].values()) == 2


def test_invalid_relation_and_empty_semantic_population_are_explicit():
    with pytest.raises(ValueError, match="unknown annotation relation"):
        summarize(np.array([17]), np.array([-1]), np.array([True]), {17: "vehicle.car"})
    result = summarize(np.array([], np.uint8), np.array([], np.uint8), np.array([], bool), {17: "vehicle.car"})
    assert all(r["events"] == 0 and r["semantic_counts"] == {"vehicle.car": 0} for r in result.values())
