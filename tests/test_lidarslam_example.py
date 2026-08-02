from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / "examples" / "lidarslam_ros2" / "dor_lidarslam.launch.py"
README = ROOT / "examples" / "lidarslam_ros2" / "README.md"
TIERS_CONFIG = ROOT / "examples" / "lidarslam_ros2" / "rko_lio_tiers_indoor02.yaml"
MANIFEST_BAG = ROOT / "scripts" / "prepare_online_manifest_rosbag.py"


def test_lidarslam_launch_is_valid_python_and_wires_cleaned_cloud():
    text = LAUNCH.read_text(encoding="utf-8")
    ast.parse(text)
    assert '"publish_deskewed_scan": True' in text
    assert '"--fixed-frame"' in text
    assert '("cloud_input", LaunchConfiguration("cleaned_topic"))' in text
    assert '("odom_input", LaunchConfiguration("cleaned_odometry_topic"))' in text
    assert '"--output-odometry-topic"' in text
    assert '"--baseline-output-topic"' in text
    assert '("map_save", "/filtered/map_save")' in text
    assert '("map_save", "/baseline/map_save")' in text
    assert '("cloud_input", LaunchConfiguration("baseline_relay_topic"))' in text
    assert 'DeclareLaunchArgument("graph_ndt_num_threads", default_value="1")' in text
    assert 'DeclareLaunchArgument("submap_distance_threshold", default_value="1.5")' in text
    assert 'DeclareLaunchArgument("run_filtered_backend", default_value="true")' in text
    assert 'LaunchConfiguration("filter_lidar_frame")' in text
    assert '"--summary-json"' in text
    assert 'LaunchConfiguration("dor_summary_json")' in text
    assert '"save_map_path": PathJoinSubstitution' in text
    assert 'DeclareLaunchArgument("use_pcd_cache", default_value="false")' in text
    assert 'DeclareLaunchArgument("frontend_start_delay", default_value="3.0")' in text
    assert 'DeclareLaunchArgument("frontend_mode", default_value="offline"' in text
    assert 'if mode == "external"' in text
    assert 'DeclareLaunchArgument("frontend_odometry_topic"' in text
    assert '"ros2", "bag", "play"' in text
    assert "TimerAction(" in text
    assert '"voxel_size": LaunchConfiguration("voxel_size")' in text


def test_lidarslam_readme_keeps_evaluation_claims_separate():
    text = README.read_text(encoding="utf-8")
    assert "online static mapping" in text
    assert "not offline map cleaning" in text
    assert "Point-count reduction alone is not" in text
    assert "Both runs report 11 pairs, 11 submaps" in text
    assert "moving-GT contamination falls 14.1%" in text
    assert "Evaluation deliberately uses `map_optimized.pcd`" in text


def test_tiers_integration_config_records_approximation():
    text = TIERS_CONFIG.read_text(encoding="utf-8")
    assert "engineering approximation" in text
    assert "not a metrology claim" in text
    assert "rko_lio_offline_node:" in text
    assert "rko_lio_online_node:" in text
    assert "ros__parameters:" in text
    assert "extrinsic_imu2base_quat_xyzw_xyz" in text


def test_manifest_bag_pose_helpers_preserve_relative_transform_and_quaternion():
    spec = importlib.util.spec_from_file_location("prepare_online_manifest_rosbag", MANIFEST_BAG)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    first_rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    second_rotation = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    relative_rotation, relative_translation = module._relative_pose(
        second_rotation,
        np.array([12.0, 23.0, 3.0]),
        first_rotation,
        np.array([10.0, 20.0, 3.0]),
    )
    np.testing.assert_allclose(relative_translation, [3.0, -2.0, 0.0], atol=1e-12)
    quaternion = module._quaternion_xyzw(relative_rotation)
    reconstructed = module._rotation_from_pose({"quaternion_xyzw": quaternion})
    np.testing.assert_allclose(reconstructed, relative_rotation, atol=1e-12)


def test_manifest_bag_prefers_integer_nanosecond_timestamp():
    spec = importlib.util.spec_from_file_location("prepare_online_manifest_rosbag", MANIFEST_BAG)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._timestamp_ns({"timestamp_ns": 315969904359876000, "timestamp_sec": 0.0}) == 315969904359876000
