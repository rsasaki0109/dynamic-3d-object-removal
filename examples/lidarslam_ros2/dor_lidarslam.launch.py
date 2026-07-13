"""Run RKO-LIO, pose-aware DOR, and the lidar_slam_ros2 map backend.

The filter is deliberately downstream of RKO-LIO: RKO-LIO supplies both a
deskewed scan and the timestamped odom TF needed by detector-free filtering.
Only the graph/map backend consumes the cleaned cloud; frontend odometry is
identical between baseline and filtered runs.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _rko_node(context):
    mode = LaunchConfiguration("frontend_mode").perform(context).strip().lower()
    if mode not in {"offline", "online"}:
        raise ValueError("frontend_mode must be 'offline' or 'online'")
    parameters = [{
        "lidar_topic": LaunchConfiguration("lidar_topic"),
        "imu_topic": LaunchConfiguration("imu_topic"),
        "base_frame": LaunchConfiguration("base_frame"),
        "odom_frame": LaunchConfiguration("fixed_frame"),
        "lidar_frame": LaunchConfiguration("lidar_frame"),
        "imu_frame": LaunchConfiguration("imu_frame"),
        "deskew": True,
        "voxel_size": LaunchConfiguration("voxel_size"),
        "max_range": LaunchConfiguration("max_range"),
        "min_range": LaunchConfiguration("min_range"),
        "initialization_phase": LaunchConfiguration("initialization_phase"),
        "publish_deskewed_scan": True,
        "deskewed_scan_topic": LaunchConfiguration("deskewed_topic"),
    }]
    if mode == "offline":
        parameters[0]["bag_path"] = LaunchConfiguration("bag_path")
    param_file = LaunchConfiguration("rko_param_file").perform(context).strip()
    if param_file:
        parameters.append(param_file)
    return [Node(
        package="rko_lio",
        executable=f"{mode}_node",
        name=f"rko_lio_{mode}_node",
        parameters=parameters,
        output="screen",
        emulate_tty=True,
    )]


def _frontend_actions(context):
    mode = LaunchConfiguration("frontend_mode").perform(context).strip().lower()
    delay = LaunchConfiguration("frontend_start_delay")
    if mode == "external":
        bag_player = ExecuteProcess(
            cmd=[
                "ros2", "bag", "play", LaunchConfiguration("bag_path"),
                "--rate", LaunchConfiguration("bag_play_rate"),
                "--topics", LaunchConfiguration("deskewed_topic"),
                LaunchConfiguration("frontend_odometry_topic"),
            ],
            output="screen",
        )
        return [TimerAction(period=delay, actions=[bag_player])]
    nodes = _rko_node(context)
    if mode == "offline":
        return [TimerAction(period=delay, actions=nodes)]
    if mode != "online":
        raise ValueError("frontend_mode must be 'offline' or 'online'")
    bag_player = ExecuteProcess(
        cmd=[
            "ros2", "bag", "play", LaunchConfiguration("bag_path"),
            "--rate", LaunchConfiguration("bag_play_rate"),
            "--topics", LaunchConfiguration("lidar_topic"), LaunchConfiguration("imu_topic"),
        ],
        output="screen",
    )
    return [*nodes, TimerAction(period=delay, actions=[bag_player])]


def _dor_process(context):
    cmd = [
        LaunchConfiguration("dor_executable"),
        "--algorithm", "range",
        "--pointcloud-topic", LaunchConfiguration("deskewed_topic"),
        "--output-topic", LaunchConfiguration("cleaned_topic"),
        "--odometry-topic", LaunchConfiguration("frontend_odometry_topic"),
        "--output-odometry-topic", LaunchConfiguration("cleaned_odometry_topic"),
        "--baseline-output-topic", LaunchConfiguration("baseline_relay_topic"),
        "--lidar-to-base",
        LaunchConfiguration("lidar_to_base_x"),
        LaunchConfiguration("lidar_to_base_y"),
        LaunchConfiguration("lidar_to_base_z"),
        "0", "0", "0", "1",
        "--queue-size", LaunchConfiguration("filter_queue_size"),
        "--fixed-frame", LaunchConfiguration("fixed_frame"),
        "--range-window", LaunchConfiguration("range_window"),
        "--range-margin", LaunchConfiguration("range_margin"),
        "--range-h-res", LaunchConfiguration("range_h_res"),
        "--range-v-res", LaunchConfiguration("range_v_res"),
        "--tf-timeout", "0.10",
        "--tf-stale-time", "0.25",
        "--expected-rate-hz", LaunchConfiguration("sensor_rate_hz"),
    ]
    summary = LaunchConfiguration("dor_summary_json").perform(context).strip()
    if summary:
        cmd.extend(["--summary-json", summary])
    return [ExecuteProcess(cmd=cmd, output="screen")]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("bag_path", description="Input rosbag2 directory or file."),
        DeclareLaunchArgument("main_param_dir", description="lidar_slam_ros2 graph backend YAML."),
        DeclareLaunchArgument("rko_param_file", default_value=""),
        DeclareLaunchArgument("lidar_topic", default_value="/os1_cloud_node1/points"),
        DeclareLaunchArgument("imu_topic", default_value="/imu/imu"),
        DeclareLaunchArgument("base_frame", default_value="base_link"),
        DeclareLaunchArgument("fixed_frame", default_value="odom"),
        DeclareLaunchArgument("lidar_frame", default_value=""),
        DeclareLaunchArgument("imu_frame", default_value=""),
        DeclareLaunchArgument("voxel_size", default_value="1.0"),
        DeclareLaunchArgument("max_range", default_value="100.0"),
        DeclareLaunchArgument("min_range", default_value="1.0"),
        DeclareLaunchArgument("initialization_phase", default_value="false"),
        DeclareLaunchArgument("frontend_mode", default_value="offline",
                              description="RKO-LIO offline/online, or external stamped cloud+odometry."),
        DeclareLaunchArgument("frontend_start_delay", default_value="3.0"),
        DeclareLaunchArgument("bag_play_rate", default_value="1.0"),
        DeclareLaunchArgument("filter_lidar_frame", default_value="sensor1/os_sensor"),
        DeclareLaunchArgument("lidar_to_base_x", default_value="0.05"),
        DeclareLaunchArgument("lidar_to_base_y", default_value="0.0"),
        DeclareLaunchArgument("lidar_to_base_z", default_value="-0.055"),
        DeclareLaunchArgument("deskewed_topic", default_value="/rko_lio/frame"),
        DeclareLaunchArgument("frontend_odometry_topic", default_value="/rko_lio/odometry"),
        DeclareLaunchArgument("cleaned_topic", default_value="/dor/cleaned_points"),
        DeclareLaunchArgument("cleaned_odometry_topic", default_value="/dor/odometry"),
        DeclareLaunchArgument("baseline_relay_topic", default_value="/dor/baseline_points"),
        DeclareLaunchArgument("filter_queue_size", default_value="512"),
        DeclareLaunchArgument("save_dir", default_value="."),
        DeclareLaunchArgument("baseline_save_dir", default_value="baseline_map"),
        DeclareLaunchArgument("run_baseline_backend", default_value="false"),
        DeclareLaunchArgument("run_filtered_backend", default_value="true"),
        DeclareLaunchArgument("use_pcd_cache", default_value="false"),
        DeclareLaunchArgument("graph_ndt_num_threads", default_value="1"),
        DeclareLaunchArgument("submap_distance_threshold", default_value="1.5"),
        DeclareLaunchArgument("dor_executable", default_value="dynamic-object-removal-realtime"),
        DeclareLaunchArgument("dor_summary_json", default_value=""),
        DeclareLaunchArgument("sensor_rate_hz", default_value="10.0"),
        DeclareLaunchArgument("range_window", default_value="3"),
        DeclareLaunchArgument("range_margin", default_value="0.5"),
        DeclareLaunchArgument("range_h_res", default_value="1.0"),
        DeclareLaunchArgument("range_v_res", default_value="2.0"),
        OpaqueFunction(function=_frontend_actions),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            arguments=[
                LaunchConfiguration("lidar_to_base_x"),
                LaunchConfiguration("lidar_to_base_y"),
                LaunchConfiguration("lidar_to_base_z"),
                "0", "0", "0", "1",
                LaunchConfiguration("base_frame"),
                LaunchConfiguration("filter_lidar_frame"),
            ],
            output="screen",
        ),
        OpaqueFunction(function=_dor_process),
        Node(
            package="graph_based_slam",
            executable="graph_based_slam_node",
            name="graph_based_slam",
            parameters=[
                LaunchConfiguration("main_param_dir"),
                {
                    "global_frame_id": "map",
                    "use_sim_time": False,
                    "use_odom_input": True,
                    "use_pcd_cache": ParameterValue(
                        LaunchConfiguration("use_pcd_cache"), value_type=bool,
                    ),
                    "ndt_num_threads": ParameterValue(
                        LaunchConfiguration("graph_ndt_num_threads"), value_type=int,
                    ),
                    "submap_distance_threshold": ParameterValue(
                        LaunchConfiguration("submap_distance_threshold"), value_type=float,
                    ),
                    "map_save_dir": LaunchConfiguration("save_dir"),
                    "save_pose_graph_path": PathJoinSubstitution([
                        LaunchConfiguration("save_dir"), "pose_graph.g2o",
                    ]),
                    "save_map_path": PathJoinSubstitution([
                        LaunchConfiguration("save_dir"), "map.pcd",
                    ]),
                },
            ],
            remappings=[
                ("odom_input", LaunchConfiguration("cleaned_odometry_topic")),
                ("cloud_input", LaunchConfiguration("cleaned_topic")),
                ("map_save", "/filtered/map_save"),
            ],
            condition=IfCondition(LaunchConfiguration("run_filtered_backend")),
            output="screen",
        ),
        Node(
            package="graph_based_slam",
            executable="graph_based_slam_node",
            name="graph_based_slam",
            parameters=[
                LaunchConfiguration("main_param_dir"),
                {
                    "global_frame_id": "map",
                    "use_sim_time": False,
                    "use_odom_input": True,
                    "use_pcd_cache": ParameterValue(
                        LaunchConfiguration("use_pcd_cache"), value_type=bool,
                    ),
                    "ndt_num_threads": ParameterValue(
                        LaunchConfiguration("graph_ndt_num_threads"), value_type=int,
                    ),
                    "submap_distance_threshold": ParameterValue(
                        LaunchConfiguration("submap_distance_threshold"), value_type=float,
                    ),
                    "map_save_dir": LaunchConfiguration("baseline_save_dir"),
                    "save_pose_graph_path": PathJoinSubstitution([
                        LaunchConfiguration("baseline_save_dir"), "pose_graph.g2o",
                    ]),
                    "save_map_path": PathJoinSubstitution([
                        LaunchConfiguration("baseline_save_dir"), "map.pcd",
                    ]),
                },
            ],
            remappings=[
                ("odom_input", LaunchConfiguration("cleaned_odometry_topic")),
                ("cloud_input", LaunchConfiguration("baseline_relay_topic")),
                ("map_save", "/baseline/map_save"),
            ],
            condition=IfCondition(LaunchConfiguration("run_baseline_backend")),
            output="screen",
        ),
    ])
