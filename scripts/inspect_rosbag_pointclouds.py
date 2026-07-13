#!/usr/bin/env python3
"""Summarize PointCloud2 timing and fields directly from a ROS 2 bag."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _stamp_seconds(stamp: object) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def inspect_bag(
    bag_path: Path, topics: set[str], imu_topics: set[str]
) -> dict[str, object]:
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import Imu
        from sensor_msgs.msg import PointCloud2
        from sensor_msgs_py import point_cloud2
    except ImportError as exc:  # pragma: no cover - depends on a sourced ROS install
        raise SystemExit("Source the ROS 2 environment before running this script") from exc

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("", ""),
    )

    samples: dict[str, list[dict[str, object]]] = defaultdict(list)
    stamps: dict[str, list[float]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    prior_imu_margins: dict[str, list[float]] = defaultdict(list)
    latest_imu_stamp: float | None = None
    while reader.has_next():
        topic, serialized, bag_stamp_ns = reader.read_next()
        if topic not in topics and topic not in imu_topics:
            continue
        if topic in imu_topics:
            msg = deserialize_message(serialized, Imu)
            counts[topic] += 1
            stamps[topic].append(_stamp_seconds(msg.header.stamp))
            latest_imu_stamp = stamps[topic][-1]
            continue
        msg = deserialize_message(serialized, PointCloud2)
        counts[topic] += 1
        stamps[topic].append(_stamp_seconds(msg.header.stamp))
        field_names = [field.name for field in msg.fields]
        sample: dict[str, object] | None = None
        if len(samples[topic]) < 2:
            sample = {
                "bag_stamp_s": bag_stamp_ns * 1e-9,
                "header_stamp_s": stamps[topic][-1],
                "frame_id": msg.header.frame_id,
                "height": msg.height,
                "width": msg.width,
                "point_step": msg.point_step,
                "fields": field_names,
            }
        time_field = next((name for name in ("t", "time", "timestamp") if name in field_names), None)
        if time_field is not None:
            points = point_cloud2.read_points(msg, field_names=[time_field])
            values = np.asarray(points[time_field]).reshape(-1)
            finite = values[np.isfinite(values)]
            if finite.size:
                point_min = float(finite.min())
                point_max = float(finite.max())
                if sample is not None:
                    sample["point_time_field"] = time_field
                    sample["point_time_min"] = point_min
                    sample["point_time_max"] = point_max
                duration_scale = 1e-9 if abs(point_max - point_min) > 100.0 else 1.0
                if latest_imu_stamp is not None:
                    scan_end = stamps[topic][-1] + point_max * duration_scale
                    prior_imu_margins[topic].append(latest_imu_stamp - scan_end)
        if sample is not None:
            samples[topic].append(sample)

    result: dict[str, object] = {}
    for topic in sorted(topics | imu_topics):
        topic_stamps = np.asarray(stamps[topic], dtype=np.float64)
        deltas = np.diff(topic_stamps)
        result[topic] = {
            "count": counts[topic],
            "header_stamp_first_s": float(topic_stamps[0]) if topic_stamps.size else None,
            "header_stamp_last_s": float(topic_stamps[-1]) if topic_stamps.size else None,
            "delta_min_s": float(deltas.min()) if deltas.size else None,
            "delta_median_s": float(np.median(deltas)) if deltas.size else None,
            "delta_max_s": float(deltas.max()) if deltas.size else None,
            "nonpositive_delta_count": int(np.count_nonzero(deltas <= 0.0)),
            "over_0_2s_delta_count": int(np.count_nonzero(deltas > 0.2)),
            "prior_imu_margin_min_s": (
                float(min(prior_imu_margins[topic])) if prior_imu_margins[topic] else None
            ),
            "prior_imu_margin_nonpositive_count": sum(
                margin <= 0.0 for margin in prior_imu_margins[topic]
            ),
            "samples": samples[topic],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag_path", type=Path)
    parser.add_argument("topics", nargs="+", help="PointCloud2 topic names")
    parser.add_argument("--imu-topic", action="append", default=[])
    args = parser.parse_args()
    print(
        json.dumps(
            inspect_bag(args.bag_path, set(args.topics), set(args.imu_topic)),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
