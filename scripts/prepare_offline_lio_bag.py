#!/usr/bin/env python3
"""Copy selected ROS 2 bag topics with a LiDAR storage-order delay.

Some offline LIO readers throttle before reading the IMU message immediately
following a full LiDAR queue. Delaying only the bag storage timestamp orders
that future IMU before the corresponding LiDAR without changing message header
timestamps or sensor payloads.
"""

from __future__ import annotations

import argparse
import heapq
import shutil
from pathlib import Path


def prepare_bag(
    input_bag: Path,
    output_bag: Path,
    lidar_topic: str,
    imu_topic: str,
    lidar_storage_delay_ms: float,
    drop_trailing_lidar: int,
) -> tuple[int, int]:
    try:
        import rosbag2_py
    except ImportError as exc:  # pragma: no cover - depends on a sourced ROS install
        raise SystemExit("Source the ROS 2 environment before running this script") from exc

    if output_bag.exists():
        shutil.rmtree(output_bag)

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(input_bag), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("", ""),
    )
    source_topics = {
        metadata.name: metadata for metadata in reader.get_all_topics_and_types()
    }
    source_counts = {
        info.topic_metadata.name: info.message_count
        for info in reader.get_metadata().topics_with_message_count
    }
    missing = {lidar_topic, imu_topic} - source_topics.keys()
    if missing:
        raise SystemExit(f"Missing requested topics: {', '.join(sorted(missing))}")
    if not 0 <= drop_trailing_lidar < source_counts[lidar_topic]:
        raise SystemExit("--drop-trailing-lidar must retain at least one LiDAR frame")

    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(output_bag), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("", ""),
    )
    for topic in (imu_topic, lidar_topic):
        writer.create_topic(source_topics[topic])

    delay_ns = round(lidar_storage_delay_ms * 1_000_000.0)
    counts = {imu_topic: 0, lidar_topic: 0}
    pending: list[tuple[int, int, str, object]] = []
    sequence = 0
    lidar_seen = 0
    while reader.has_next():
        topic, serialized, storage_stamp_ns = reader.read_next()
        if topic not in counts:
            continue
        if topic == lidar_topic:
            lidar_seen += 1
            if lidar_seen > source_counts[lidar_topic] - drop_trailing_lidar:
                continue
            storage_stamp_ns += delay_ns
        heapq.heappush(
            pending, (storage_stamp_ns, sequence, topic, serialized)
        )
        sequence += 1
        counts[topic] += 1
        # Future source messages cannot precede the current unshifted storage
        # time, so everything due by now can be written in adjusted order.
        while pending and pending[0][0] <= storage_stamp_ns - (delay_ns if topic == lidar_topic else 0):
            adjusted_stamp_ns, _, due_topic, due_serialized = heapq.heappop(pending)
            writer.write(due_topic, due_serialized, adjusted_stamp_ns)
    while pending:
        adjusted_stamp_ns, _, due_topic, due_serialized = heapq.heappop(pending)
        writer.write(due_topic, due_serialized, adjusted_stamp_ns)
    return counts[lidar_topic], counts[imu_topic]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_bag", type=Path)
    parser.add_argument("output_bag", type=Path)
    parser.add_argument("--lidar-topic", required=True)
    parser.add_argument("--imu-topic", required=True)
    parser.add_argument("--lidar-storage-delay-ms", type=float, default=20.0)
    parser.add_argument("--drop-trailing-lidar", type=int, default=0)
    args = parser.parse_args()
    lidar_count, imu_count = prepare_bag(
        args.input_bag,
        args.output_bag,
        args.lidar_topic,
        args.imu_topic,
        args.lidar_storage_delay_ms,
        args.drop_trailing_lidar,
    )
    print(f"wrote {lidar_count} LiDAR and {imu_count} IMU messages to {args.output_bag}")


if __name__ == "__main__":
    main()
