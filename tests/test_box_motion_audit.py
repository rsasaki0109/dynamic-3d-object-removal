import numpy as np
import pytest

from scripts.audit_nuscenes_box_motion import barrier_height_context, motion_summary, pose_contract, target_memberships


def row(sample, center, length=4., token="track"):
    return {"token": token + sample, "sample_token": sample, "instance_token": token,
            "translation": [center, 0, 0], "size": [1., length, 1.], "rotation": [1, 0, 0, 0],
            "attribute_tokens": ["parked"], "num_lidar_pts": 10}


SAMPLES = {"first": {"timestamp": 0}, "last": {"timestamp": 1500000}}
ATTRIBUTES = {"parked": {"name": "vehicle.parked"}}


def test_extent_change_can_trigger_center_rule_while_one_projected_face_stays_fixed():
    result = motion_summary([row("first", 0, 4), row("last", 3, 10)], SAMPLES, ATTRIBUTES)
    assert result["moving_by_frozen_center_rule"] is True
    assert result["max_center_displacement_m"] == 3
    assert result["projected_bound_spans_uvz_m"][:2] == [0, 6]
    assert result["size_min_wlh_m"] == [1, 4, 1]
    assert result["size_max_wlh_m"] == [1, 10, 1]
    assert result["attribute_observation_counts"] == {"vehicle.parked": 2}


def test_rigid_translation_moves_both_faces_and_metadata_order_does_not_change_anchor():
    result = motion_summary([row("last", 3), row("first", 0)], SAMPLES, ATTRIBUTES)
    assert result["observations"][0]["sample_token"] == "first"
    assert result["projected_bound_spans_uvz_m"][:2] == [3, 3]
    assert result["max_annotation_center_step_rate_m_per_s"] == 2
    boundary = motion_summary([row("first", 0), row("last", 2)], SAMPLES, ATTRIBUTES)
    assert boundary["moving_by_frozen_center_rule"] is False


def test_single_observation_and_duplicate_times_are_not_invented_motion():
    single = motion_summary([row("first", 0)], SAMPLES, ATTRIBUTES)
    assert single["moving_by_frozen_center_rule"] is False
    assert single["max_annotation_center_step_rate_m_per_s"] is None
    with pytest.raises(ValueError, match="distinct increasing"):
        motion_summary([row("first", 0), row("first", 4)], SAMPLES, ATTRIBUTES)


def test_overlapping_moving_boxes_retain_multiple_memberships_without_losing_population():
    annotations = {"first": [row("first", 0, token="a"), row("first", 0, token="b")]}
    tokens, pairs, track, multiplicity = target_memberships(np.array([[0., 0, 0], [10., 0, 0]]),
        np.array([0, 0]), np.array([1, 0], bool), annotations, ["first"], {"a", "b"}, .25)
    assert tokens == ["a", "b"]
    assert pairs.tolist() == [0, 0] and track.tolist() == [0, 1]
    assert multiplicity.tolist() == [2, 0]
    assert int((multiplicity > 0).sum()) == 1


def test_uncovered_target_or_duplicate_track_annotation_fails():
    points = np.array([[0., 0, 0]])
    with pytest.raises(AssertionError):
        target_memberships(points, np.array([0]), np.array([True]), {"first": []}, ["first"], {"a"}, .25)
    duplicate = {"first": [row("first", 0, token="a"), row("first", 0, token="a")]}
    with pytest.raises(ValueError, match="duplicate instance"):
        target_memberships(points, np.array([0]), np.array([True]), duplicate, ["first"], {"a"}, .25)


def test_common_vertical_shift_triggers_original_rule_but_not_horizontal_counterfactual():
    first, last = row("first", 0), row("last", 0)
    last["translation"][2] = 3
    origins = {"first": [0, 0, 1], "last": [0, 0, 4]}
    result = motion_summary([first, last], SAMPLES, ATTRIBUTES, sensor_origins=origins)
    assert result["moving_by_frozen_center_rule"] is True
    assert result["horizontal_diagnostic_would_mark_moving"] is False
    assert result["max_vertical_displacement_m"] == result["sensor_origin_z_span_m"] == 3
    assert result["center_minus_sensor_z_span_m"] == 0


def test_barrier_height_context_ignores_box_z_and_excludes_other_semantics():
    annotation = row("first", 0)
    annotation["translation"][2] = 100.
    points = np.array([[0., 0, 1.], [1., 0, 2.], [0., 0, -50.], [10., 0, 99.]])
    result = barrier_height_context(points, np.array([9, 9, 17, 9]), annotation, 9, .25)
    assert result["barrier_points_in_xy_footprint"] == 2
    assert result["barrier_z_quantiles_m"] == [1, 1.25, 1.5, 1.75, 2]


def test_pose_contract_reports_raw_heights_without_assuming_a_zero_value():
    result = pose_contract({"ego": {"a": {"translation": [0, 0, 0]}, "b": {"translation": [1, 2, 0]}}})
    assert result["ego_pose_translation_z_values_m"] == [0]
    assert result["selected_ego_pose_records"] == 2
    changed = pose_contract({"ego": {"a": {"translation": [0, 0, 3]}}})
    assert changed["ego_pose_translation_z_values_m"] == [3]
