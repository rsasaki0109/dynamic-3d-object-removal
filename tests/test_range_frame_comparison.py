import numpy as np

import dynamic_object_removal as core
from scripts.compare_range_frames import pose_rotation, sensor_frame_keep


PARAMS = {"h_res_deg": 2.5, "v_res_deg": 2.5, "range_margin": .5,
          "min_see_through": 2, "max_surface_hits": 1, "ground_z": None,
          "resolutions": None}


def test_sensor_frame_votes_follow_pose_and_preserve_map_row_order():
    local_map = np.array([[5., 0., 1.], [10., 0., 2.], [0., 5., 1.]])
    local_query = local_map[1:]
    angle = np.deg2rad(31)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    origin = np.array([100., -50., 2.])
    global_map = local_map @ rotation.T + origin
    local_scans = [(local_query, rotation, origin)] * 3
    _, expected = core.clean_map_by_visibility(local_map, [(local_query, np.zeros(3))] * 3, **PARAMS)
    assert expected.tolist() == [False, True, True]
    np.testing.assert_array_equal(sensor_frame_keep(global_map, local_scans, PARAMS), expected)
    # Ground protection belongs to the map frame, not sensor-frame Z.
    params = {**PARAMS, "ground_z": 3.1}
    assert sensor_frame_keep(global_map, local_scans, params).tolist() == [True, True, True]


def test_quaternion_and_matrix_pose_rotations_agree():
    angle = np.deg2rad(31)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    q = [0, 0, np.sin(angle / 2), np.cos(angle / 2)]
    np.testing.assert_allclose(pose_rotation({"pose": {"quaternion_xyzw": q}}), rotation)
    np.testing.assert_array_equal(pose_rotation({"pose": {"rotation": rotation.tolist()}}), rotation)
