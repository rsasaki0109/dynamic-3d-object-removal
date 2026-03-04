#!/usr/bin/env python3
"""Realtime dynamic object removal node."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import dynamic_object_removal as core


def _eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def _now_sec() -> float:
    return time.monotonic()


def _to_float(value: Any) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError("not a number")
    return float(value)


def _first_value(obj: Any, *names: str) -> Any | None:
    for name in names:
        if isinstance(obj, Mapping):
            if name in obj:
                return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _as_vec3(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        x = _first_value(value, "x")
        y = _first_value(value, "y")
        z = _first_value(value, "z")
        if x is not None and y is not None and z is not None:
            try:
                return np.array([_to_float(x), _to_float(y), _to_float(z)], dtype=np.float64)
            except Exception:
                return None
        return None
    if not isinstance(value, str):
        if hasattr(value, "x") and hasattr(value, "y") and hasattr(value, "z"):
            try:
                return np.array([
                    _to_float(value.x),
                    _to_float(value.y),
                    _to_float(value.z),
                ], dtype=np.float64)
            except Exception:
                return None
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            arr = np.asarray(value, dtype=np.float64).reshape(-1)
            if arr.size == 3:
                return arr.astype(np.float64)
    return None


def _as_quat(value: Any) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    x = _first_value(value, "x")
    y = _first_value(value, "y")
    z = _first_value(value, "z")
    w = _first_value(value, "w")
    if None in (x, y, z, w):
        return None
    try:
        return (_to_float(x), _to_float(y), _to_float(z), _to_float(w))
    except Exception:
        return None


def _quat_yaw(value: Any) -> float | None:
    quat = _as_quat(value)
    if quat is None:
        return None
    x, y, z, w = quat
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _stamp_to_sec(stamp: Any) -> float | None:
    if stamp is None:
        return None
    if isinstance(stamp, (int, float)):
        return float(stamp)
    if hasattr(stamp, "to_sec"):
        try:
            return float(stamp.to_sec())
        except Exception:
            pass
    if hasattr(stamp, "sec") and hasattr(stamp, "nanosec"):
        try:
            return float(stamp.sec) + float(stamp.nanosec) * 1e-9
        except Exception:
            pass
    if hasattr(stamp, "secs") and hasattr(stamp, "nsecs"):
        try:
            return float(stamp.secs) + float(stamp.nsecs) * 1e-9
        except Exception:
            pass
    if isinstance(stamp, Mapping):
        if "sec" in stamp:
            try:
                sec = _to_float(stamp["sec"])
                nsec = _to_float(stamp.get("nanosec", 0.0))
                return sec + nsec * 1e-9
            except Exception:
                pass
        if "secs" in stamp:
            try:
                sec = _to_float(stamp["secs"])
                nsec = _to_float(stamp.get("nsecs", 0.0))
                return sec + nsec * 1e-9
            except Exception:
                pass
    return None


def _extract_msg_stamp(msg: Any) -> float | None:
    if msg is None:
        return None
    for target in (msg, _first_value(msg, "header"), _first_value(msg, "msg_header")):
        if target is None:
            continue
        ts = _stamp_to_sec(_first_value(target, "stamp"))
        if ts is not None:
            return ts
    return _stamp_to_sec(_first_value(msg, "stamp", "time", "timestamp"))


def _extract_box_entry_center(entry: Any) -> np.ndarray | None:
    center_like = _first_value(entry, "center", "position")
    if center_like is not None:
        if isinstance(center_like, Mapping) and "position" in center_like:
            center_like = center_like["position"]
        vec = _as_vec3(center_like)
        if vec is not None:
            return vec

    bbox = _first_value(entry, "bbox", "box", "bounding_box")
    if bbox is not None:
        bbox_center = _first_value(bbox, "center", "position")
        if isinstance(bbox_center, Mapping) and "position" in bbox_center:
            bbox_center = bbox_center["position"]
        if bbox_center is None:
            pose = _first_value(bbox, "pose")
            if pose is not None:
                bbox_center = _first_value(pose, "position")
        vec = _as_vec3(bbox_center)
        if vec is not None:
            return vec

    if isinstance(entry, Mapping) and all(k in entry for k in ("x", "y", "z")):
        vec = _as_vec3(entry)
        if vec is not None:
            return vec

    pose = _first_value(entry, "pose")
    if pose is not None:
        position = _first_value(pose, "position")
        if position is not None:
            vec = _as_vec3(position)
            if vec is not None:
                return vec

    return None


def _extract_box_entry_size(entry: Any) -> np.ndarray | None:
    for key in ("size", "dimensions", "extent", "scale", "lwh", "shape"):
        value = _first_value(entry, key)
        vec = _as_vec3(value)
        if vec is not None and np.all(vec > 0.0):
            return vec

    bbox = _first_value(entry, "bbox", "box", "bounding_box")
    if bbox is not None:
        bbox_size = _first_value(bbox, "size", "dimensions", "extent", "scale", "lwh", "shape")
        vec = _as_vec3(bbox_size)
        if vec is not None and np.all(vec > 0.0):
            return vec

    if isinstance(entry, Mapping):
        if all(k in entry for k in ("length", "width", "height")):
            try:
                return np.array(
                    [_to_float(entry["length"]), _to_float(entry["width"]), _to_float(entry["height"])],
                    dtype=np.float64,
                )
            except Exception:
                pass
        if all(k in entry for k in ("l", "w", "h")):
            try:
                return np.array([_to_float(entry["l"]), _to_float(entry["w"]), _to_float(entry["h"])], dtype=np.float64)
            except Exception:
                pass

    return None


def _extract_box_entry_yaw(entry: Any) -> float | None:
    yaw = _first_value(entry, "yaw", "heading")
    if yaw is not None:
        try:
            return _to_float(yaw)
        except Exception:
            pass

    yaw_deg = _first_value(entry, "yaw_deg", "heading_deg")
    if yaw_deg is not None:
        try:
            return math.radians(_to_float(yaw_deg))
        except Exception:
            pass

    orientation = _first_value(entry, "orientation", "rotation")
    yaw = _quat_yaw(orientation)
    if yaw is not None:
        return yaw

    pose = _first_value(entry, "pose")
    orientation = _first_value(pose, "orientation") if pose is not None else None
    yaw = _quat_yaw(orientation)
    if yaw is not None:
        return yaw

    bbox = _first_value(entry, "bbox", "box", "bounding_box")
    if bbox is not None:
        orientation = _first_value(bbox, "orientation", "rotation")
        yaw = _quat_yaw(orientation)
        if yaw is not None:
            return yaw
        pose = _first_value(bbox, "pose")
        orientation = _first_value(pose, "orientation")
        yaw = _quat_yaw(orientation)
        if yaw is not None:
            return yaw

    rpy = _first_value(entry, "rpy")
    if isinstance(rpy, Sequence) and not isinstance(rpy, (str, bytes)) and len(rpy) >= 3:
        try:
            return _to_float(rpy[2])
        except Exception:
            pass

    return None


def _extract_box_entry_label(entry: Any) -> str | None:
    value = _first_value(entry, "label", "name", "class_name", "class_id", "object_type", "type")
    if isinstance(value, str):
        return value
    return None


def _normalize_box_entry(entry: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {}
    if isinstance(entry, Mapping):
        raw.update(entry)

    center = _extract_box_entry_center(entry)
    if center is not None:
        raw.setdefault("center", center.tolist())
    size = _extract_box_entry_size(entry)
    if size is not None:
        raw.setdefault("size", size.tolist())

    yaw = _extract_box_entry_yaw(entry)
    if yaw is not None:
        raw.setdefault("yaw", float(yaw))

    label = _extract_box_entry_label(entry)
    if label is not None:
        raw.setdefault("label", label)

    return raw


def _extract_box_candidates(payload: Any) -> tuple[list[Any], float | None]:
    stamp = _extract_msg_stamp(payload)
    if isinstance(payload, str):
        data = json.loads(payload)
        detected_stamp = _extract_msg_stamp(data) or stamp
        candidates, _ = _extract_box_candidates(data)
        return candidates, detected_stamp

    if isinstance(payload, Mapping):
        for key in ("objects", "detections", "boxes", "targets", "results"):
            value = payload.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return list(value), stamp
        return [payload], stamp

    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        return list(payload), stamp

    # Generic object-like ROS messages that expose list fields as attributes
    for key in ("objects", "detections", "boxes", "targets", "results"):
        value = _first_value(payload, key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return list(value), stamp

    return [payload], stamp


def parse_object_message(payload: Any, *, skip_invalid: bool) -> tuple[list[core.DetectionBox], float | None]:
    candidates, stamp = _extract_box_candidates(payload)
    boxes: list[core.DetectionBox] = []
    for item in candidates:
        raw = _normalize_box_entry(item)
        if "center" not in raw or "size" not in raw:
            if skip_invalid:
                _eprint(f"skip invalid box item: {item}")
                continue
            raise ValueError(f"invalid box item: {item}")
        boxes.extend(core.parse_boxes_payload([raw], skip_invalid=skip_invalid))
    return boxes, stamp


def _load_ros_message_class(spec: str) -> Any:
    module_name, sep, class_name = spec.rpartition(".")
    if not sep:
        raise ValueError(f"invalid ROS message type: {spec}")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise ValueError(f"unknown ROS message type: {spec}") from exc


def _ros_imports() -> dict[str, Any]:
    import rclpy
    from rclpy.node import Node

    from std_msgs.msg import String
    from sensor_msgs.msg import PointCloud2, PointField

    return {
        "rclpy": rclpy,
        "Node": Node,
        "String": String,
        "PointCloud2": PointCloud2,
        "PointField": PointField,
    }


def _point_field_dtype_code(point_field_type: int, endian: str, point_field: Any) -> str:
    mapping = {
        point_field.INT8: "i1",
        point_field.UINT8: "u1",
        point_field.INT16: "i2",
        point_field.UINT16: "u2",
        point_field.INT32: "i4",
        point_field.UINT32: "u4",
        point_field.FLOAT32: "f4",
        point_field.FLOAT64: "f8",
    }
    if point_field_type not in mapping:
        raise ValueError(f"unsupported PointField type: {point_field_type}")
    return f"{endian}{mapping[point_field_type]}"


def _point_cloud2_to_xyz(msg: Any, PointCloud2: Any, PointField: Any) -> np.ndarray:
    if not isinstance(msg, PointCloud2):
        raise TypeError("expected PointCloud2 message")
    if msg.point_step <= 0 or msg.width <= 0 or msg.height <= 0:
        return np.zeros((0, 3), dtype=np.float64)

    endian = ">" if bool(msg.is_bigendian) else "<"
    needed = ("x", "y", "z")
    names: list[str] = []
    formats: list[str] = []
    offsets: list[int] = []

    for field in msg.fields:
        if field.name not in needed:
            continue
        if field.count != 1:
            raise ValueError(f"unsupported field count for {field.name}: {field.count}")
        names.append(field.name)
        formats.append(_point_field_dtype_code(int(field.datatype), endian, PointField))
        offsets.append(int(field.offset))

    if len(names) != 3 or set(names) != set(needed):
        raise ValueError("PointCloud2 must contain x,y,z scalar float fields")

    dtype = np.dtype(
        {
            "names": names,
            "formats": formats,
            "offsets": offsets,
            "itemsize": int(msg.point_step),
        }
    )
    count = int(msg.width) * int(msg.height)
    raw = np.frombuffer(bytes(msg.data), dtype=dtype, count=count)
    if raw.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    points = np.column_stack((raw["x"], raw["y"], raw["z"]))
    return np.asarray(points, dtype=np.float64).reshape(-1, 3)


def _xyz_to_point_cloud2(points: np.ndarray, template: Any, PointCloud2: Any, PointField: Any) -> Any:
    if points.size == 0:
        points = points.reshape(0, 3)
    points_f32 = np.asarray(points, dtype=np.float32).reshape(-1, 3)

    output = PointCloud2()
    output.header = template.header
    output.height = 1
    output.width = int(points_f32.shape[0])
    output.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    output.is_bigendian = False
    output.point_step = 12
    output.row_step = 12 * output.width
    output.is_dense = True
    output.data = points_f32.tobytes()
    return output


def _percentile(values: Sequence[float], percent: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, percent))


def _summarize_times(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "max_ms": float(arr.max()),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Realtime dynamic object remover node for PointCloud2."
    )
    parser.add_argument(
        "--pointcloud-topic",
        default="/points_raw",
        help="Input sensor_msgs/PointCloud2 topic.",
    )
    parser.add_argument(
        "--objects-topic",
        default="/dynamic_objects",
        help="Input topic containing object detections.",
    )
    parser.add_argument(
        "--objects-msg-type",
        default="std_msgs.msg.String",
        help="ROS message type for object topic. Example: std_msgs.msg.String, vision_msgs.msg.Detection3DArray.",
    )
    parser.add_argument(
        "--output-topic",
        default="/points_static",
        help="Output sensor_msgs/PointCloud2 topic.",
    )
    parser.add_argument(
        "--algorithm",
        choices=["box", "temporal"],
        default="box",
        help="Filter mode: box (requires detection input) or temporal.",
    )
    parser.add_argument(
        "--box-margin",
        nargs=3,
        type=float,
        default=list(core.DEFAULT_BOX_MARGIN),
        metavar=("X", "Y", "Z"),
        help="Safety margin around each detection box (meters).",
    )
    parser.add_argument("--min-size", type=float, default=0.01, help="Ignore boxes smaller than this in any axis.")
    parser.add_argument("--skip-invalid", action="store_true", help="Skip invalid detection rows.")
    parser.add_argument(
        "--box-stale-time",
        type=float,
        default=0.25,
        help="Ignore boxes older than this (seconds).",
    )
    parser.add_argument("--max-object-history", type=int, default=64, help="Maximum cached object messages.")
    parser.add_argument("--voxel-size", type=float, default=core.DEFAULT_TEMPORAL_VOXEL_SIZE, help="Temporal filter voxel size.")
    parser.add_argument("--temporal-window", type=int, default=5, help="Temporal filter window size.")
    parser.add_argument("--temporal-min-hits", type=int, default=3, help="Temporal filter minimum hits.")
    parser.add_argument("--queue-size", type=int, default=20, help="ROS subscription queue size.")
    parser.add_argument("--stats-period", type=int, default=100, help="Log summary every N frames.")
    parser.add_argument("--quiet", action="store_true", help="Reduce logs.")
    parser.add_argument("--summary-json", help="Save runtime summary JSON at shutdown.")
    parser.add_argument("--version", action="version", version="dynamic-object-removal-realtime 0.1.0")
    return parser


@dataclass
class _RealtimeStats:
    frames: int = 0
    input_points: int = 0
    output_points: int = 0
    removed_points: int = 0
    boxes_received: int = 0
    frames_with_boxes: int = 0
    frames_without_boxes: int = 0
    stale_box_frames: int = 0
    callback_ms: deque[float] | None = None

    def __post_init__(self) -> None:
        self.callback_ms = deque(maxlen=4096)

    def update(
        self,
        in_count: int,
        out_count: int,
        duration_ms: float,
        *,
        used_boxes: bool,
        no_boxes: bool,
    ) -> None:
        self.frames += 1
        self.input_points += in_count
        self.output_points += out_count
        self.removed_points += max(0, in_count - out_count)
        if used_boxes:
            self.frames_with_boxes += 1
        if no_boxes:
            self.frames_without_boxes += 1
            self.stale_box_frames += 1
        if self.callback_ms is not None:
            self.callback_ms.append(duration_ms)

    def summary(self) -> dict[str, Any]:
        times = _summarize_times(self.callback_ms or [])
        return {
            "frames": self.frames,
            "input_points": self.input_points,
            "output_points": self.output_points,
            "removed_points": self.removed_points,
            "removed_ratio": (self.removed_points / self.input_points) if self.input_points else 0.0,
            "boxes_received": self.boxes_received,
            "frames_with_boxes": self.frames_with_boxes,
            "frames_without_boxes": self.frames_without_boxes,
            "stale_box_frames": self.stale_box_frames,
            **times,
        }


@dataclass
class _TimedBoxes:
    boxes: list[core.DetectionBox]
    msg_stamp: float | None
    received: float


class DynamicObjectRemovalNode:
    def __init__(self, **kwargs: Any) -> None:
        imports = _ros_imports()
        self._rclpy = imports["rclpy"]
        Node = imports["Node"]
        self._PointCloud2 = imports["PointCloud2"]
        PointField = imports["PointField"]
        self._String = imports["String"]

        self._pointcloud_topic = str(kwargs["pointcloud_topic"])
        self._objects_topic = str(kwargs["objects_topic"])
        self._output_topic = str(kwargs["output_topic"])
        self._algorithm = str(kwargs["algorithm"])
        self._box_margin = [float(v) for v in kwargs["box_margin"]]
        self._min_size = float(kwargs["min_size"])
        self._skip_invalid = bool(kwargs["skip_invalid"])
        self._box_stale_time = float(kwargs["box_stale_time"])
        self._stats_period = max(1, int(kwargs["stats_period"]))
        self._quiet = bool(kwargs["quiet"])
        self._summary_json = kwargs.get("summary_json")
        self._queue_size = int(kwargs["queue_size"])
        self._max_object_history = max(1, int(kwargs["max_object_history"]))
        self._objects_msg_type = str(kwargs["objects_msg_type"])
        self._pointfield = PointField

        self._timed_boxes: deque[_TimedBoxes] = deque(maxlen=self._max_object_history)
        self._stats = _RealtimeStats()

        self._temporal_filter = None
        if self._algorithm == "temporal":
            self._temporal_filter = core.TemporalConsistencyFilter(
                voxel_size=float(kwargs["voxel_size"]),
                window_size=int(kwargs["temporal_window"]),
                min_hits=int(kwargs["temporal_min_hits"]),
            )

        try:
            self._objects_msg_type_class = _load_ros_message_class(self._objects_msg_type)
        except Exception as exc:
            raise RuntimeError(f"failed to load objects message type '{self._objects_msg_type}': {exc}") from exc

        self._node = Node("dynamic_object_removal_realtime")
        self._sub_pc = self._node.create_subscription(
            self._PointCloud2, self._pointcloud_topic, self._on_pointcloud, self._queue_size
        )
        self._sub_objects = None
        if self._algorithm == "box":
            self._sub_objects = self._node.create_subscription(
                self._objects_msg_type_class,
                self._objects_topic,
                self._on_objects,
                self._queue_size,
            )
        self._pub_pc = self._node.create_publisher(self._PointCloud2, self._output_topic, self._queue_size)

        self._node.get_logger().info(
            f"dynamic-object-removal-realtime started: algorithm={self._algorithm}, "
            f"pointcloud={self._pointcloud_topic}, output={self._output_topic}"
        )
        if self._algorithm == "box":
            self._node.get_logger().info(
                f"boxes from {self._objects_topic} type={self._objects_msg_type} stale={self._box_stale_time}s "
                f"history={self._max_object_history}"
            )

    def _select_boxes(self, point_stamp: float | None) -> tuple[list[core.DetectionBox], bool, bool]:
        if not self._timed_boxes:
            return [], False, True

        now = _now_sec()
        while (
            self._timed_boxes
            and self._timed_boxes[0].msg_stamp is None
            and (now - self._timed_boxes[0].received) > self._box_stale_time
        ):
            self._timed_boxes.popleft()

        if not self._timed_boxes:
            return [], False, True

        if point_stamp is not None:
            while (
                self._timed_boxes
                and self._timed_boxes[0].msg_stamp is not None
                and point_stamp - self._timed_boxes[0].msg_stamp > self._box_stale_time
            ):
                self._timed_boxes.popleft()
            if self._timed_boxes:
                best: _TimedBoxes | None = None
                best_diff = self._box_stale_time
                has_stamped = False
                for entry in self._timed_boxes:
                    if entry.msg_stamp is None:
                        continue
                    has_stamped = True
                    diff = abs(point_stamp - entry.msg_stamp)
                    if diff <= self._box_stale_time and diff < best_diff:
                        best = entry
                        best_diff = diff
                if best is not None:
                    return best.boxes, bool(best.boxes), False

                if has_stamped:
                    return [], False, True

                latest = self._timed_boxes[-1]
                if now - latest.received > self._box_stale_time:
                    return [], False, True

                # use latest recent cache when no exact stamp match
                if now - latest.received <= self._box_stale_time:
                    return latest.boxes, bool(latest.boxes), False
                return [], False, True

            return [], False, True

        # Without point timestamp, use latest recent cache in receive-time window.
        latest = self._timed_boxes[-1]
        if latest.msg_stamp is None and now - latest.received > self._box_stale_time:
            return [], False, True
        return latest.boxes, bool(latest.boxes), False

    def _on_objects(self, msg: Any) -> None:
        try:
            payload = msg.data if isinstance(msg, self._String) else msg
            boxes, msg_stamp = parse_object_message(payload, skip_invalid=self._skip_invalid)
            boxes = [b for b in boxes if (b.size >= self._min_size).all()]
            if not boxes:
                self._node.get_logger().warn("received no valid boxes")

            if msg_stamp is None:
                msg_stamp = _extract_msg_stamp(msg)

            self._timed_boxes.append(
                _TimedBoxes(
                    boxes=boxes,
                    msg_stamp=msg_stamp,
                    received=_now_sec(),
                )
            )
            self._stats.boxes_received += 1
            if not self._quiet:
                self._node.get_logger().info(f"received {len(boxes)} boxes")
        except Exception as exc:
            self._node.get_logger().warn(f"invalid box message: {exc}")

    def _on_pointcloud(self, msg: Any) -> None:
        start = _now_sec()
        try:
            points = _point_cloud2_to_xyz(msg, self._PointCloud2, self._pointfield)
            filtered = points
            if points.size == 0:
                filtered = points
                used_boxes = False
                stale = False

            elif self._algorithm == "box":
                point_stamp = _extract_msg_stamp(msg)
                boxes, _, stale = self._select_boxes(point_stamp)
                if not stale and boxes:
                    filtered = core.remove_points_in_boxes(points, boxes, self._box_margin)[0]
                    used_boxes = True
                elif not stale and not boxes:
                    filtered = points
                    used_boxes = False
                else:
                    filtered = points
                    used_boxes = False

            else:
                assert self._temporal_filter is not None
                filtered = self._temporal_filter.filter(points)[0]
                used_boxes = False
                stale = False

            self._publish(msg, filtered)

            duration_ms = (_now_sec() - start) * 1000.0
            self._stats.update(
                len(points),
                len(filtered),
                duration_ms,
                used_boxes=used_boxes and not stale,
                no_boxes=stale or not used_boxes,
            )
            if not self._quiet and self._stats.frames % self._stats_period == 0:
                summary = self._stats.summary()
                self._node.get_logger().info(
                    "frames={} in={} out={} removed={:.2f}% p95={}ms boxes_with_msg={} stale_frames={}".format(
                        summary["frames"],
                        summary["input_points"],
                        summary["output_points"],
                        summary["removed_ratio"] * 100.0,
                        _percentile(self._stats.callback_ms or [], 95) if self._stats.callback_ms else 0.0,
                        summary["frames_with_boxes"],
                        summary["stale_box_frames"],
                    )
                )
        except Exception as exc:
            self._node.get_logger().error(f"pointcloud processing failed: {exc}")

    def _publish(self, template: Any, points: np.ndarray) -> None:
        out_msg = _xyz_to_point_cloud2(points, template, self._PointCloud2, self._pointfield)
        self._pub_pc.publish(out_msg)

    def write_summary(self) -> None:
        if not self._summary_json:
            return
        Path(self._summary_json).write_text(
            json.dumps(self._stats.summary(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def destroy(self) -> None:
        self.write_summary()
        self._node.destroy_node()

    def spin(self) -> None:
        self._rclpy.spin(self._node)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        node = DynamicObjectRemovalNode(
            pointcloud_topic=args.pointcloud_topic,
            objects_topic=args.objects_topic,
            objects_msg_type=args.objects_msg_type,
            output_topic=args.output_topic,
            algorithm=args.algorithm,
            box_margin=args.box_margin,
            min_size=args.min_size,
            skip_invalid=args.skip_invalid,
            box_stale_time=args.box_stale_time,
            max_object_history=args.max_object_history,
            voxel_size=args.voxel_size,
            temporal_window=args.temporal_window,
            temporal_min_hits=args.temporal_min_hits,
            queue_size=args.queue_size,
            stats_period=args.stats_period,
            quiet=args.quiet,
            summary_json=args.summary_json,
        )
        try:
            node.spin()
            return 0
        finally:
            node.destroy()
    except Exception as exc:
        _eprint(f"failed to start realtime node: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
