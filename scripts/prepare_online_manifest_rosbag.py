#!/usr/bin/env python3
"""Convert an online benchmark manifest into a stamped ROS 2 cloud/odometry bag.

The manifest remains the source of truth for point labels and poses. The output bag
contains only real sensor-frame points plus odometry; labels are deliberately not
published to the cleaner. Poses are rebased to the first frame to keep map coordinates
numerically well-conditioned while preserving every relative transform.

Run under a sourced ROS 2 environment::

    source /opt/ros/jazzy/setup.bash
    python3 scripts/prepare_online_manifest_rosbag.py manifest.json /tmp/av2_rosbag
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _rotation_from_pose(payload: dict[str, Any]) -> np.ndarray:
    if "rotation" in payload:
        rotation = np.asarray(payload["rotation"], dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError("pose.rotation must have shape (3,3)")
        return rotation
    q = np.asarray(payload.get("quaternion_xyzw"), dtype=np.float64)
    if q.shape != (4,):
        raise ValueError("pose needs rotation or quaternion_xyzw")
    x, y, z, w = q
    norm = float(np.linalg.norm(q))
    if norm == 0.0:
        raise ValueError("pose quaternion cannot be zero")
    x, y, z, w = q / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    """Convert a proper rotation matrix to a normalized xyzw quaternion."""
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("rotation must have shape (3,3)")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        q = np.array([
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
            0.25 * scale,
        ])
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            q = np.array([0.25 * scale, (matrix[0, 1] + matrix[1, 0]) / scale,
                          (matrix[0, 2] + matrix[2, 0]) / scale,
                          (matrix[2, 1] - matrix[1, 2]) / scale])
        elif axis == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            q = np.array([(matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale,
                          (matrix[1, 2] + matrix[2, 1]) / scale,
                          (matrix[0, 2] - matrix[2, 0]) / scale])
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            q = np.array([(matrix[0, 2] + matrix[2, 0]) / scale,
                          (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale,
                          (matrix[1, 0] - matrix[0, 1]) / scale])
    return q / np.linalg.norm(q)


def _timestamp_ns(frame: dict[str, Any]) -> int:
    if "timestamp_ns" in frame:
        stamp = int(frame["timestamp_ns"])
    elif "timestamp_sec" in frame:
        stamp = int(round(float(frame["timestamp_sec"]) * 1e9))
    else:
        raise ValueError("frame has no timestamp_ns/timestamp_sec")
    if stamp < 0:
        raise ValueError("frame timestamp must be non-negative")
    return stamp


def _relative_pose(
    rotation: np.ndarray,
    translation: np.ndarray,
    first_rotation: np.ndarray,
    first_translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return first_rotation.T @ rotation, first_rotation.T @ (translation - first_translation)


def convert_manifest(
    manifest_path: Path,
    output_uri: Path,
    *,
    cloud_topic: str,
    odometry_topic: str,
    lidar_frame: str,
    fixed_frame: str,
    base_frame: str,
) -> dict[str, Any]:
    try:
        import rosbag2_py
        from nav_msgs.msg import Odometry
        from rclpy.serialization import serialize_message
        from sensor_msgs_py import point_cloud2
        from std_msgs.msg import Header
    except ImportError as exc:
        raise SystemExit("source a ROS 2 environment before running this script") from exc

    manifest_path = manifest_path.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("manifest.frames must be a non-empty list")
    if output_uri.exists():
        raise FileExistsError(f"output already exists: {output_uri}")

    first_pose = frames[0].get("pose")
    if not isinstance(first_pose, dict):
        raise ValueError("first frame has no pose")
    first_rotation = _rotation_from_pose(first_pose)
    first_translation = np.asarray(first_pose.get("translation"), dtype=np.float64)
    if first_translation.shape != (3,):
        raise ValueError("pose.translation must have shape (3,)")

    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(output_uri), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("", ""),
    )
    writer.create_topic(rosbag2_py.TopicMetadata(
        id=0, name=odometry_topic, type="nav_msgs/msg/Odometry",
        serialization_format="cdr", offered_qos_profiles=[],
    ))
    writer.create_topic(rosbag2_py.TopicMetadata(
        id=1, name=cloud_topic, type="sensor_msgs/msg/PointCloud2",
        serialization_format="cdr", offered_qos_profiles=[],
    ))

    previous_stamp = -1
    total_points = 0
    for index, frame in enumerate(frames):
        stamp_ns = _timestamp_ns(frame)
        if stamp_ns <= previous_stamp:
            raise ValueError(f"frame {index} timestamp is not strictly increasing")
        previous_stamp = stamp_ns
        pose = frame.get("pose")
        if not isinstance(pose, dict):
            raise ValueError(f"frame {index} has no pose")
        rotation = _rotation_from_pose(pose)
        translation = np.asarray(pose.get("translation"), dtype=np.float64)
        if translation.shape != (3,):
            raise ValueError(f"frame {index} pose.translation must have shape (3,)")
        rel_rotation, rel_translation = _relative_pose(
            rotation, translation, first_rotation, first_translation,
        )
        quaternion = _quaternion_xyzw(rel_rotation)

        cloud_path = (manifest_path.parent / frame["cloud"]).resolve()
        points = np.asarray(np.load(cloud_path), dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(f"frame {index} cloud must have shape (N,3+)")
        points = np.ascontiguousarray(points[:, :3])
        total_points += len(points)

        sec, nanosec = divmod(stamp_ns, 1_000_000_000)
        header = Header()
        header.stamp.sec = int(sec)
        header.stamp.nanosec = int(nanosec)
        header.frame_id = lidar_frame

        odom = Odometry()
        odom.header = header
        odom.header.frame_id = fixed_frame
        odom.child_frame_id = base_frame
        odom.pose.pose.position.x = float(rel_translation[0])
        odom.pose.pose.position.y = float(rel_translation[1])
        odom.pose.pose.position.z = float(rel_translation[2])
        odom.pose.pose.orientation.x = float(quaternion[0])
        odom.pose.pose.orientation.y = float(quaternion[1])
        odom.pose.pose.orientation.z = float(quaternion[2])
        odom.pose.pose.orientation.w = float(quaternion[3])
        cloud = point_cloud2.create_cloud_xyz32(header, points)

        # Exact-stamp odometry first makes replay deterministic; the realtime node also
        # supports the opposite arrival order through its bounded pair cache.
        writer.write(odometry_topic, serialize_message(odom), stamp_ns)
        writer.write(cloud_topic, serialize_message(cloud), stamp_ns)

    summary = {
        "source_manifest": str(manifest_path),
        "dataset": payload.get("dataset"),
        "scene": payload.get("scene"),
        "frames": len(frames),
        "points": total_points,
        "cloud_topic": cloud_topic,
        "odometry_topic": odometry_topic,
        "lidar_frame": lidar_frame,
        "fixed_frame": fixed_frame,
        "base_frame": base_frame,
        "poses_rebased_to_first_frame": True,
        "labels_published_to_cleaner": False,
    }
    (output_uri / "dor_source.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_uri", type=Path)
    parser.add_argument("--cloud-topic", default="/av2/points")
    parser.add_argument("--odometry-topic", default="/av2/odometry")
    parser.add_argument("--lidar-frame", default="lidar")
    parser.add_argument("--fixed-frame", default="odom")
    parser.add_argument("--base-frame", default="base_link")
    args = parser.parse_args(argv)
    summary = convert_manifest(
        args.manifest,
        args.output_uri,
        cloud_topic=args.cloud_topic,
        odometry_topic=args.odometry_topic,
        lidar_frame=args.lidar_frame,
        fixed_frame=args.fixed_frame,
        base_frame=args.base_frame,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
