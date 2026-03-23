#!/usr/bin/env python3
"""Dynamic object removal CLI.

This tool removes points that lie inside detected 3D bounding boxes.
It follows the same practical idea as the `dynamic_object_removal` ROS node:
crop each incoming point cloud by detected object boxes and keep only static points.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections import deque
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from typing import Any, Sequence

import numpy as np


DEFAULT_BOX_MARGIN = (0.05, 0.05, 0.05)
DEFAULT_TEMPORAL_VOXEL_SIZE = 0.10


@dataclass(frozen=True)
class DetectionBox:
    center: np.ndarray  # [x, y, z]
    size: np.ndarray  # [length, width, height]
    yaw: float = 0.0
    label: str | None = None


def _eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def _to_float(value: Any) -> float:
    if value is None:
        raise ValueError("missing numeric value")
    if isinstance(value, bool):
        raise ValueError("not a number")
    return float(value)


def _as_vec3(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        if all(k in value for k in ("x", "y", "z")):
            return np.array([_to_float(value["x"]), _to_float(value["y"]), _to_float(value["z"])], dtype=np.float64)
        raise ValueError("dict does not have x,y,z")
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size != 3:
            raise ValueError("expect length 3 vector")
        return arr.astype(np.float64)
    raise ValueError("invalid vector format")


def _yaw_from_quaternion(quat: Any) -> float:
    if isinstance(quat, dict):
        x = _to_float(quat.get("x", 0.0))
        y = _to_float(quat.get("y", 0.0))
        z = _to_float(quat.get("z", 0.0))
        w = _to_float(quat.get("w", 1.0))
    else:
        if not isinstance(quat, Sequence):
            raise ValueError("invalid quaternion")
        if len(quat) != 4:
            raise ValueError("invalid quaternion length")
        x, y, z, w = map(_to_float, quat)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _extract_box_center(raw: dict[str, Any]) -> np.ndarray:
    if "center" in raw:
        return _as_vec3(raw["center"])
    if all(k in raw for k in ("x", "y", "z")):
        return np.array([_to_float(raw["x"]), _to_float(raw["y"]), _to_float(raw["z"])], dtype=np.float64)
    if "position" in raw:
        return _as_vec3(raw["position"])
    if "pose" in raw and isinstance(raw["pose"], dict):
        pose = raw["pose"]
        if "position" in pose:
            return _as_vec3(pose["position"])
    raise ValueError("cannot parse box center")


def _extract_box_size(raw: dict[str, Any]) -> np.ndarray:
    size_key_candidates: list[str] = ["size", "dimensions", "extent", "bbox", "box"]
    for key in size_key_candidates:
        if key in raw:
            vec = _as_vec3(raw[key])
            if np.all(vec > 0.0):
                return vec

    if all(k in raw for k in ("length", "width", "height")):
        return np.array(
            [_to_float(raw["length"]), _to_float(raw["width"]), _to_float(raw["height"])],
            dtype=np.float64,
        )
    if all(k in raw for k in ("l", "w", "h")):
        return np.array([_to_float(raw["l"]), _to_float(raw["w"]), _to_float(raw["h"])], dtype=np.float64)

    if "shape" in raw and isinstance(raw["shape"], dict):
        shape = raw["shape"]
        if all(k in shape for k in ("x", "y", "z")):
            return _as_vec3(shape)

    raise ValueError("cannot parse box size")


def _extract_box_yaw(raw: dict[str, Any]) -> float:
    if "yaw" in raw:
        return _to_float(raw["yaw"])
    if "yaw_deg" in raw:
        return math.radians(_to_float(raw["yaw_deg"]))
    if "heading" in raw:
        return _to_float(raw["heading"])
    if "orientation" in raw:
        return _yaw_from_quaternion(raw["orientation"])
    if "rotation" in raw:
        rot = raw["rotation"]
        if isinstance(rot, dict):
            if "yaw" in rot:
                return _to_float(rot["yaw"])
            if "z" in rot:
                return _to_float(rot["z"])
        if isinstance(rot, Sequence) and not isinstance(rot, (str, bytes)):
            return _to_float(rot[2]) if len(rot) >= 3 else 0.0
    if "rpy" in raw and isinstance(raw["rpy"], Sequence):
        rpy = raw["rpy"]
        if len(rpy) >= 3:
            return _to_float(rpy[2])
    return 0.0


def _parse_box_entry(entry: Any) -> DetectionBox:
    if not isinstance(entry, dict):
        raise ValueError("box entry must be an object")
    center = _extract_box_center(entry)
    size = _extract_box_size(entry)
    yaw = _extract_box_yaw(entry)
    label = entry.get("label") if isinstance(entry.get("label"), str) else None
    return DetectionBox(center=center, size=size, yaw=yaw, label=label)


def _normalize_box_payload(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        if isinstance(raw.get("objects"), list):
            return raw["objects"]
        if isinstance(raw.get("detections"), list):
            return raw["detections"]
        if isinstance(raw.get("boxes"), list):
            return raw["boxes"]
        return [raw]
    if isinstance(raw, list):
        return raw
    raise ValueError("unsupported box payload format")


def parse_boxes_payload(raw: Any, *, skip_invalid: bool) -> list[DetectionBox]:
    boxes: list[DetectionBox] = []
    for item in _normalize_box_payload(raw):
        try:
            box = _parse_box_entry(item)
            boxes.append(box)
        except Exception as exc:
            if skip_invalid:
                _eprint(f"skip invalid box entry: {exc}")
                continue
            raise ValueError(f"invalid box entry: {item}") from exc
    return boxes


def _load_boxes_from_json(path: Path, *, skip_invalid: bool) -> list[DetectionBox]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return parse_boxes_payload(data, skip_invalid=skip_invalid)


def _read_row_as_float(row: Sequence[str]) -> list[float]:
    try:
        return [_to_float(x) for x in row]
    except Exception as exc:
        raise ValueError("invalid numeric row") from exc


def _load_boxes_from_csv(path: Path, *, skip_invalid: bool) -> list[DetectionBox]:
    boxes: list[DetectionBox] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                raw: dict[str, float | str] = {}
                row_map = {k.lower().strip(): v for k, v in row.items() if k}

                def get_float(*keys: str) -> float:
                    for key in keys:
                        if key in row_map and row_map[key] != "":
                            return _to_float(row_map[key])
                    raise ValueError("missing field")

                x = get_float("x", "cx", "center_x", "px", "pose_x")
                y = get_float("y", "cy", "center_y", "py", "pose_y")
                z = get_float("z", "cz", "center_z", "pz", "pose_z")

                l = get_float("length", "size_x", "l", "dx")
                w = get_float("width", "size_y", "w", "dy")
                h = get_float("height", "size_z", "h", "dz")

                yaw = 0.0
                for key in ("yaw", "heading", "theta", "rz"):
                    if key in row_map:
                        yaw = _to_float(row_map[key])
                        if "deg" in key:
                            yaw = math.radians(yaw)
                        break
                if "yaw_deg" in row_map:
                    yaw = math.radians(_to_float(row_map["yaw_deg"]))

                row_norm = {
                    "center": [x, y, z],
                    "size": [l, w, h],
                    "yaw": yaw,
                    "label": row_map.get("label"),
                }
                boxes.append(_parse_box_entry(row_norm))
            except Exception as exc:
                if skip_invalid:
                    _eprint(f"skip invalid row: {exc}")
                    continue
                raise ValueError(f"invalid row in {path}: {row}") from exc
    return boxes


_KITTI_DYNAMIC_CLASSES = {"Car", "Van", "Truck", "Pedestrian", "Cyclist", "Person_sitting", "Tram"}


def _parse_kitti_calib(calib_path: Path) -> np.ndarray:
    """Parse KITTI calibration file and return 4x4 cam-to-velo transform."""
    with calib_path.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith("Tr_velo_to_cam:") or line.startswith("Tr_velo_cam"):
                vals = [float(x) for x in line.split(":")[1].split()]
                T = np.eye(4)
                T[:3, :] = np.array(vals).reshape(3, 4)
                return np.linalg.inv(T)
    raise ValueError(f"Tr_velo_to_cam not found in {calib_path}")


def _load_boxes_from_kitti(
    label_path: Path,
    *,
    calib_path: Path | None = None,
    skip_invalid: bool = False,
) -> list[DetectionBox]:
    """Load KITTI label_2 format boxes, converting from camera to velodyne frame."""
    cam_to_velo: np.ndarray | None = None
    if calib_path is not None:
        cam_to_velo = _parse_kitti_calib(calib_path)

    boxes: list[DetectionBox] = []
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return boxes

    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 15:
            if skip_invalid:
                continue
            raise ValueError(f"KITTI label line has fewer than 15 fields: {line}")
        obj_type = parts[0]
        if obj_type not in _KITTI_DYNAMIC_CLASSES:
            continue
        try:
            h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
            x_cam, y_cam, z_cam = float(parts[11]), float(parts[12]), float(parts[13])
            ry = float(parts[14])
        except (ValueError, IndexError) as exc:
            if skip_invalid:
                _eprint(f"skip invalid KITTI label: {exc}")
                continue
            raise

        # KITTI location is bottom-center in camera frame; move to 3D center
        y_cam_center = y_cam - h / 2.0
        cam_pt = np.array([x_cam, y_cam_center, z_cam, 1.0])

        if cam_to_velo is not None:
            velo_pt = cam_to_velo @ cam_pt
            R = cam_to_velo[:3, :3]
            dir_cam = np.array([math.sin(ry), 0.0, math.cos(ry)])
            dir_velo = R @ dir_cam
            yaw_velo = math.atan2(dir_velo[1], dir_velo[0])
        else:
            # Without calibration: use approximate KITTI default transform
            # cam(x_right, y_down, z_forward) -> velo(x_forward, y_left, z_up)
            velo_pt = np.array([z_cam, -x_cam, -(y_cam_center), 1.0])
            yaw_velo = -(ry + math.pi / 2.0)

        box = DetectionBox(
            center=np.array([velo_pt[0], velo_pt[1], velo_pt[2]]),
            size=np.array([l, w, h]),
            yaw=yaw_velo,
            label=obj_type,
        )
        boxes.append(box)
    return boxes


def _load_boxes_from_av2_feather(
    path: Path,
    *,
    timestamp_ns: int | None = None,
    skip_invalid: bool = False,
) -> list[DetectionBox]:
    """Load Argoverse 2 annotations.feather as DetectionBox list."""
    try:
        import pyarrow.feather as feather
    except ImportError as exc:
        raise ImportError("pyarrow is required to load .feather files: pip install pyarrow") from exc

    table = feather.read_table(path)
    required = {"tx_m", "ty_m", "tz_m", "length_m", "width_m", "height_m", "qw", "qx", "qy", "qz"}
    if not required.issubset(set(table.column_names)):
        raise ValueError(f"feather file missing required AV2 annotation columns: {path}")

    if "timestamp_ns" in table.column_names:
        n_timestamps = len(set(table["timestamp_ns"].to_pylist()))
        if timestamp_ns is not None:
            ts_arr = table["timestamp_ns"].to_numpy()
            mask = ts_arr == timestamp_ns
            table = table.filter(mask)
        elif n_timestamps > 1:
            _eprint(f"warning: AV2 annotations contain {n_timestamps} timestamps; use --timestamp-ns to filter to a single frame")

    boxes: list[DetectionBox] = []
    for i in range(table.num_rows):
        try:
            tx = float(table["tx_m"][i].as_py())
            ty = float(table["ty_m"][i].as_py())
            tz = float(table["tz_m"][i].as_py())
            l = float(table["length_m"][i].as_py())
            w = float(table["width_m"][i].as_py())
            h = float(table["height_m"][i].as_py())
            qw = float(table["qw"][i].as_py())
            qx = float(table["qx"][i].as_py())
            qy = float(table["qy"][i].as_py())
            qz = float(table["qz"][i].as_py())
            yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            label = table["category"][i].as_py() if "category" in table.column_names else None
            boxes.append(DetectionBox(
                center=np.array([tx, ty, tz]),
                size=np.array([l, w, h]),
                yaw=yaw,
                label=label,
            ))
        except Exception as exc:
            if skip_invalid:
                _eprint(f"skip invalid AV2 annotation row {i}: {exc}")
                continue
            raise
    return boxes


def load_boxes(
    path: Path,
    *,
    fmt: str,
    skip_invalid: bool,
    calib_path: Path | None = None,
    timestamp_ns: int | None = None,
) -> list[DetectionBox]:
    fmt = fmt.lower()
    if fmt == "auto":
        if path.suffix.lower() in {".json", ".jsn"}:
            fmt = "json"
        elif path.suffix.lower() == ".feather":
            fmt = "av2"
        elif path.suffix.lower() in {".csv", ".tsv", ".txt"}:
            fmt = "csv"
        else:
            fmt = "json"
    if fmt == "json":
        return _load_boxes_from_json(path, skip_invalid=skip_invalid)
    if fmt == "csv":
        return _load_boxes_from_csv(path, skip_invalid=skip_invalid)
    if fmt == "kitti":
        return _load_boxes_from_kitti(path, calib_path=calib_path, skip_invalid=skip_invalid)
    if fmt == "av2":
        return _load_boxes_from_av2_feather(path, timestamp_ns=timestamp_ns, skip_invalid=skip_invalid)
    raise ValueError(f"unsupported box format: {fmt}")


def _load_ascii_point_cloud(path: Path, delimiter: str | None = None) -> np.ndarray:
    data = np.loadtxt(path, delimiter=delimiter, comments="#", dtype=np.float64)
    if data.ndim == 1:
        data = data[None, :]
    if data.size == 0:
        return data.reshape(0, 3)
    if data.shape[1] < 3:
        raise ValueError(f"point cloud must contain at least 3 columns: {path}")
    return data[:, :3]


def _load_points_csv_or_txt(path: Path) -> np.ndarray:
    first_line: str | None = None
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            first_line = line
            break
    if first_line is None:
        return np.zeros((0, 3), dtype=np.float64)

    delimiter = "," if "," in first_line else None
    first_tokens = [x.strip() for x in first_line.split(delimiter or " ")]
    has_header = False
    try:
        float(first_tokens[0])
    except Exception:
        has_header = True
    except IndexError:
        has_header = True

    if has_header:
        with path.open(encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        header = [h.strip().lower() for h in lines[0].replace(",", " ").split() if h.strip()]
        lower = {name: i for i, name in enumerate(header)}

        def idx(*names: str) -> int:
            for n in names:
                if n in lower:
                    return lower[n]
            return -1

        xi = idx("x", "px", "point_x", "position_x")
        yi = idx("y", "py", "point_y", "position_y")
        zi = idx("z", "pz", "point_z", "position_z")
        if min(xi, yi, zi) < 0:
            xi, yi, zi = 0, 1, 2

        data = np.loadtxt(lines[1:], delimiter=delimiter)
        if data.ndim == 1:
            data = data[None, :]
        if data.shape[1] <= max(xi, yi, zi):
            raise ValueError("not enough numeric columns")
        return data[:, [xi, yi, zi]]
    else:
        return _load_ascii_point_cloud(path, delimiter=delimiter)


def _pcd_scalar_dtype(type_code: str, size: int) -> np.dtype:
    low = type_code.lower()
    if low == "f" and size in {4, 8}:
        return np.dtype(f"<f{size}")
    if low == "i" and size in {1, 2, 4, 8}:
        return np.dtype(f"<i{size}")
    if low == "u" and size in {1, 2, 4, 8}:
        return np.dtype(f"<u{size}")
    raise ValueError(f"unsupported PCD field type: type={type_code} size={size}")


def _load_pcd(path: Path) -> np.ndarray:
    fields: list[str] = []
    sizes: list[int] = []
    types: list[str] = []
    counts: list[int] = []
    points = 0
    data_kind = ""
    payload = b""

    with path.open("rb") as f:
        while True:
            raw = f.readline()
            if not raw:
                break
            line = raw.decode("ascii", errors="strict").strip()
            if not line or line.startswith("#"):
                continue
            low = line.lower()
            if low.startswith("fields "):
                fields = [token.lower() for token in line.split()[1:]]
            elif low.startswith("size "):
                sizes = [int(token) for token in line.split()[1:]]
            elif low.startswith("type "):
                types = [token.upper() for token in line.split()[1:]]
            elif low.startswith("count "):
                counts = [int(token) for token in line.split()[1:]]
            elif low.startswith("points "):
                points = int(line.split()[1])
            elif low.startswith("data "):
                data_kind = low.split()[1]
                payload = f.read()
                break

    if not fields:
        raise ValueError("PCD header missing FIELDS line")
    if not data_kind:
        raise ValueError("PCD header missing DATA section")
    if not sizes or len(sizes) != len(fields):
        raise ValueError("PCD SIZE does not match FIELDS")
    if not types or len(types) != len(fields):
        raise ValueError("PCD TYPE does not match FIELDS")
    if not counts:
        counts = [1] * len(fields)
    if len(counts) != len(fields):
        raise ValueError("PCD COUNT does not match FIELDS")
    if points < 0:
        raise ValueError("PCD POINTS must be non-negative")

    idx = {k: fields.index(k) for k in ("x", "y", "z") if k in fields}
    if len(idx) != 3:
        raise ValueError("PCD FIELDS must include x,y,z")
    if data_kind == "binary_compressed":
        raise ValueError("PCD DATA binary_compressed is not supported")

    if data_kind == "ascii":
        point_lines = [ln for ln in payload.decode("utf-8").splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
        if not point_lines:
            return np.zeros((0, 3), dtype=np.float64)
        data = np.loadtxt(io.StringIO("\n".join(point_lines)), dtype=np.float64)
        if data.ndim == 1:
            data = data[None, :]
        if data.shape[1] < len(fields):
            raise ValueError("PCD point format is shorter than expected")
        return data[:, [idx["x"], idx["y"], idx["z"]]]

    if data_kind != "binary":
        raise ValueError(f"unsupported PCD DATA type: {data_kind}")

    dtype_fields: list[tuple[Any, ...]] = []
    for name, size, type_code, count in zip(fields, sizes, types, counts):
        scalar_dtype = _pcd_scalar_dtype(type_code, size)
        if count == 1:
            dtype_fields.append((name, scalar_dtype))
        else:
            dtype_fields.append((name, scalar_dtype, (count,)))
    point_dtype = np.dtype(dtype_fields)
    expected_size = point_dtype.itemsize * points
    if len(payload) < expected_size:
        raise ValueError("PCD binary payload is shorter than expected")
    data = np.frombuffer(payload[:expected_size], dtype=point_dtype, count=points)
    return np.column_stack(
        (
            np.asarray(data["x"], dtype=np.float64),
            np.asarray(data["y"], dtype=np.float64),
            np.asarray(data["z"], dtype=np.float64),
        )
    )


def _load_kitti_bin(path: Path) -> np.ndarray:
    """Load KITTI velodyne .bin file (float32 x4: x, y, z, reflectance)."""
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 4 != 0:
        raise ValueError(f"KITTI .bin file size not divisible by 4: {path}")
    points = raw.reshape(-1, 4)
    return points[:, :3].astype(np.float64)


def _load_feather_points(path: Path) -> np.ndarray:
    """Load point cloud from Apache Feather file (Argoverse 2 format)."""
    try:
        import pyarrow.feather as feather
    except ImportError as exc:
        raise ImportError("pyarrow is required to load .feather files: pip install pyarrow") from exc
    table = feather.read_table(path)
    if not all(c in table.column_names for c in ("x", "y", "z")):
        raise ValueError(f"feather file must have x,y,z columns: {path}")
    return np.column_stack([
        table["x"].to_numpy(zero_copy_only=False),
        table["y"].to_numpy(zero_copy_only=False),
        table["z"].to_numpy(zero_copy_only=False),
    ]).astype(np.float64)


def load_points(path: Path, *, fmt: str) -> np.ndarray:
    fmt = fmt.lower()
    if fmt == "auto":
        if path.suffix.lower() == ".npy":
            fmt = "npy"
        elif path.suffix.lower() == ".pcd":
            fmt = "pcd"
        elif path.suffix.lower() == ".bin":
            fmt = "bin"
        elif path.suffix.lower() == ".feather":
            fmt = "feather"
        elif path.suffix.lower() in {".csv", ".txt", ".xyz", ".pts", ".tsv"}:
            fmt = "text"
        else:
            fmt = "text"
    if fmt == "npy":
        arr = np.load(path)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[1] < 3:
            raise ValueError("npy cloud must have at least 3 columns")
        return np.asarray(arr[:, :3], dtype=np.float64)
    if fmt == "pcd":
        return _load_pcd(path)
    if fmt == "bin":
        return _load_kitti_bin(path)
    if fmt == "feather":
        return _load_feather_points(path)
    if fmt == "text":
        return _load_points_csv_or_txt(path)
    raise ValueError(f"unsupported cloud format: {fmt}")


@dataclass
class TemporalConsistencyFilter:
    voxel_size: float = DEFAULT_TEMPORAL_VOXEL_SIZE
    window_size: int = 5
    min_hits: int = 3

    def __post_init__(self) -> None:
        if self.voxel_size <= 0.0:
            raise ValueError("voxel_size must be positive")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.min_hits <= 0:
            raise ValueError("min_hits must be positive")
        self._history: deque[set[tuple[int, int, int]]] = deque(maxlen=self.window_size)
        self._voxel_hits: Counter[tuple[int, int, int]] = Counter()

    def filter(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if points.size == 0 or len(points) == 0:
            return points, np.ones(0, dtype=bool)

        voxels = np.floor(points / self.voxel_size).astype(np.int64)
        frame_voxels = {tuple(v) for v in np.unique(voxels, axis=0)}

        if self._history.maxlen and len(self._history) >= self._history.maxlen:
            old_frame = self._history.popleft()
            for voxel in old_frame:
                self._voxel_hits[voxel] -= 1
                if self._voxel_hits[voxel] <= 0:
                    del self._voxel_hits[voxel]

        self._history.append(frame_voxels)
        for voxel in frame_voxels:
            self._voxel_hits[voxel] += 1

        point_voxels = [tuple(v) for v in voxels]
        keep_mask = np.fromiter(
            (self._voxel_hits[voxel] >= self.min_hits for voxel in point_voxels),
            dtype=bool,
            count=points.shape[0],
        )
        return points[keep_mask], keep_mask


def _rotate_by_yaw(points: np.ndarray, yaw: float) -> np.ndarray:
    if points.size == 0 or abs(yaw) < 1e-12:
        return points
    c = math.cos(-yaw)
    s = math.sin(-yaw)
    rotated = np.empty_like(points)
    rotated[:, 0] = c * points[:, 0] - s * points[:, 1]
    rotated[:, 1] = s * points[:, 0] + c * points[:, 1]
    rotated[:, 2] = points[:, 2]
    return rotated


def remove_points_in_boxes(points: np.ndarray, boxes: Sequence[DetectionBox], margin: Sequence[float] = (0.05, 0.05, 0.05)) -> tuple[np.ndarray, np.ndarray]:
    if points.size == 0 or len(points) == 0:
        return points, np.ones(0, dtype=bool)
    if len(boxes) == 0:
        mask = np.ones(points.shape[0], dtype=bool)
        return points, mask

    m = np.asarray(margin, dtype=np.float64)
    if m.shape != (3,):
        raise ValueError("margin must have 3 elements")

    keep = np.ones(points.shape[0], dtype=bool)
    xyz = np.asarray(points, dtype=np.float64)
    for box in boxes:
        half = box.size * 0.5 + m
        if np.any(half <= 0.0):
            continue
        local = xyz[keep] - box.center
        local = _rotate_by_yaw(local, box.yaw)
        inside = (
            (local[:, 0] >= -half[0])
            & (local[:, 0] <= half[0])
            & (local[:, 1] >= -half[1])
            & (local[:, 1] <= half[1])
            & (local[:, 2] >= -half[2])
            & (local[:, 2] <= half[2])
        )
        keep_idx = np.nonzero(keep)[0]
        mask_local = np.ones_like(keep)
        mask_local[keep_idx] = ~inside
        keep &= mask_local
    return xyz[keep], keep


def _save_pcd_ascii(path: Path, points: np.ndarray) -> None:
    n = points.shape[0]
    lines: list[str] = [
        "VERSION .7",
        "FIELDS x y z",
        "TYPE F F F",
        "SIZE 4 4 4",
        "COUNT 1 1 1",
        f"WIDTH {n}",
        "HEIGHT 1",
        f"POINTS {n}",
        "DATA ascii",
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")
        for row in points:
            f.write(f"{row[0]:.10f} {row[1]:.10f} {row[2]:.10f}\n")


def save_points(path: Path, points: np.ndarray, *, fmt: str) -> None:
    if fmt == "auto":
        fmt = path.suffix.lower().lstrip(".")
    fmt = fmt.lower()
    if fmt == "pcd":
        _save_pcd_ascii(path, points)
        return
    if fmt in {"npy", "npz"}:
        np.save(path, points)
        return
    delimiter = "," if fmt == "csv" else " "
    header = "x,y,z" if fmt == "csv" else "x y z"
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{header}\n")
        writer = csv.writer(f, delimiter=delimiter)
        for x, y, z in points:
            writer.writerow([f"{x:.10f}", f"{y:.10f}", f"{z:.10f}"])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Remove points inside detection boxes from point clouds.")
    parser.add_argument("--input-cloud", required=True, help="Input point cloud path (csv/txt/xyz/pcd/npy).")
    parser.add_argument("--input-objects", required=True, help="Detected object boxes JSON or CSV path.")
    parser.add_argument("--output-cloud", required=True, help="Output point cloud path.")
    parser.add_argument("--cloud-format", default="auto", choices=["auto", "csv", "pcd", "text", "npy", "bin", "feather"], help="Output/input point cloud format.")
    parser.add_argument("--objects-format", default="auto", choices=["auto", "json", "csv", "kitti", "av2"], help="Object file format.")
    parser.add_argument("--calib-path", default=None, help="KITTI calibration file path (required when --objects-format=kitti).")
    parser.add_argument("--timestamp-ns", type=int, default=None, help="Filter AV2 annotations by timestamp (nanoseconds).")
    parser.add_argument("--box-margin", nargs=3, type=float, default=list(DEFAULT_BOX_MARGIN), metavar=("X", "Y", "Z"), help="Safety margin around each box (meters).")
    parser.add_argument("--skip-invalid", action="store_true", help="Skip invalid object entries instead of stopping.")
    parser.add_argument("--min-size", type=float, default=0.01, help="Skip boxes smaller than this size in any axis.")
    parser.add_argument("--summary-json", help="Write filtering statistics as JSON to this path.")
    parser.add_argument("--quiet", action="store_true", help="Suppress stdout summary.")
    parser.add_argument("--version", action="version", version="dynamic-object-removal 0.1.0")
    return parser


def _filter_small_boxes(boxes: Sequence[DetectionBox], min_size: float) -> list[DetectionBox]:
    if min_size <= 0.0:
        return list(boxes)
    return [b for b in boxes if (b.size >= min_size).all()]


def _write_summary_json(path: Path, *, total: int, kept: int, boxes: Sequence[DetectionBox], removed: int) -> None:
    payload = {
        "total_points": int(total),
        "kept_points": int(kept),
        "removed_points": int(removed),
        "removed_ratio": float(removed / total) if total else 0.0,
        "objects": [
            {
                "label": box.label,
                "center": box.center.tolist(),
                "size": box.size.tolist(),
                "yaw": box.yaw,
            }
            for box in boxes
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    cloud_path = Path(args.input_cloud)
    obj_path = Path(args.input_objects)
    out_path = Path(args.output_cloud)

    if not cloud_path.exists():
        _eprint(f"input cloud not found: {cloud_path}")
        return 1
    if not obj_path.exists():
        _eprint(f"object file not found: {obj_path}")
        return 1

    calib = Path(args.calib_path) if args.calib_path else None
    boxes = load_boxes(
        obj_path,
        fmt=args.objects_format,
        skip_invalid=args.skip_invalid,
        calib_path=calib,
        timestamp_ns=args.timestamp_ns,
    )
    boxes = _filter_small_boxes(boxes, args.min_size)

    if not boxes:
        _eprint("no valid boxes. nothing will be removed.")

    points = load_points(cloud_path, fmt=args.cloud_format)
    filtered, keep_mask = remove_points_in_boxes(points, boxes, args.box_margin)

    removed = points.shape[0] - filtered.shape[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_points(out_path, filtered, fmt=args.cloud_format)

    if not args.quiet:
        ratio = 0.0 if points.shape[0] == 0 else removed / points.shape[0]
        _eprint(f"input: {points.shape[0]} points")
        _eprint(f"objects: {len(boxes)}")
        _eprint(f"removed: {removed} points ({ratio:.2%})")
        _eprint(f"output: {filtered.shape[0]} points -> {out_path}")

    if args.summary_json:
        _write_summary_json(
            Path(args.summary_json),
            total=points.shape[0],
            kept=filtered.shape[0],
            boxes=boxes,
            removed=removed,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
