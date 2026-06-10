#!/usr/bin/env python3
"""Dynamic object removal for LiDAR point clouds (numpy-only, no deep learning).

Five geometric algorithms in one small module: ``box`` (crop by detected 3D boxes),
``temporal`` (voxel hit-consistency over a window), ``range`` (range-image visibility,
Removert-style remove + revert), ``scan_ratio`` (per-column pseudo-occupancy,
ERASOR-style), and ``fusion`` (free-space carving + DUFOMap-style eroded voids +
scan-ratio votes, OR-fused). All but ``box`` are detector-free map cleaners.
No GPU, no model.
"""

from __future__ import annotations

__version__ = "0.5.0"

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

# Range-image (visibility) ghost removal defaults.
DEFAULT_RANGE_H_RES_DEG = 0.4
DEFAULT_RANGE_V_RES_DEG = 1.0
DEFAULT_RANGE_MARGIN = 0.5

# Scan-ratio (ERASOR R-POD) pseudo-occupancy removal defaults.
DEFAULT_SR_RINGS = 20
DEFAULT_SR_SECTORS = 108
DEFAULT_SR_MAX_RANGE = 80.0
DEFAULT_SR_RATIO = 0.2
DEFAULT_SR_MIN_MAP_HEIGHT = 0.5
DEFAULT_SR_GROUND_MARGIN = 0.2
DEFAULT_SR_VOTES_FRACTION = 0.5  # majority of the scans that actually revisit a point's column
DEFAULT_SR_VOTES_FLOOR = 3

# Free-space fusion (``fusion``) defaults: three OR-ed evidence channels.
DEFAULT_FUSION_MIN_RANGE = 1.0
DEFAULT_FUSION_MAX_RANGE = 60.0
# Channel 1 — plain free-space carving (ray sampling, hit-precedence).
DEFAULT_FREE_VOXEL = 0.3
DEFAULT_FREE_STEP = 0.3
DEFAULT_FREE_CARVE_MARGIN = 0.6
DEFAULT_FREE_GROUND_MARGIN = 0.25
DEFAULT_FREE_VOTES_FRACTION = 0.9
DEFAULT_FREE_VOTES_FLOOR = 2
# Channel 2 — eroded void carving (DUFOMap-style d_s hit inflation + d_p erosion).
DEFAULT_VOID_VOXEL = 0.2
DEFAULT_VOID_STEP = 0.1
DEFAULT_VOID_HIT_INFLATION = 0.2
DEFAULT_VOID_MIN_SCANS = 11
# Channel 3 — scan-ratio votes reused at a stricter fraction than the standalone default.
DEFAULT_FUSION_SR_FRACTION = 0.7


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


@dataclass(frozen=True)
class PcdScan:
    """One PCD scan with optional per-point intensity and sensor pose (VIEWPOINT)."""

    points: np.ndarray  # (N, 3)
    intensity: np.ndarray | None = None  # (N,)
    viewpoint: np.ndarray | None = None  # (7,) tx, ty, tz, qw, qx, qy, qz


def _parse_pcd_header(header_lines: list[str]) -> tuple[dict[str, Any], str, bytes]:
    fields: list[str] = []
    sizes: list[int] = []
    types: list[str] = []
    counts: list[int] = []
    points = 0
    width = 0
    height = 1
    viewpoint: list[float] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    data_kind = ""
    payload = b""

    for line in header_lines:
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
        elif low.startswith("width "):
            width = int(line.split()[1])
        elif low.startswith("height "):
            height = int(line.split()[1])
        elif low.startswith("points "):
            points = int(line.split()[1])
        elif low.startswith("viewpoint "):
            viewpoint = [float(token) for token in line.split()[1:]]
        elif low.startswith("data "):
            data_kind = low.split()[1]
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
    if len(viewpoint) != 7:
        raise ValueError("PCD VIEWPOINT must have 7 values: tx ty tz qw qx qy qz")

    metadata = {
        "fields": fields,
        "sizes": sizes,
        "types": types,
        "counts": counts,
        "points": points,
        "width": width,
        "height": height,
        "viewpoint": viewpoint,
        "data_kind": data_kind,
    }
    return metadata, data_kind, payload


def _pcd_structured_dtype(metadata: dict[str, Any]) -> np.dtype:
    dtype_fields: list[tuple[Any, ...]] = []
    for name, size, type_code, count in zip(
        metadata["fields"], metadata["sizes"], metadata["types"], metadata["counts"]
    ):
        scalar_dtype = _pcd_scalar_dtype(type_code, size)
        if count == 1:
            dtype_fields.append((name, scalar_dtype))
        else:
            dtype_fields.append((name, scalar_dtype, (count,)))
    return np.dtype(dtype_fields)


def _structured_pcd_to_scan(data: np.ndarray, metadata: dict[str, Any]) -> PcdScan:
    fields = metadata["fields"]
    if not all(k in fields for k in ("x", "y", "z")):
        raise ValueError("PCD FIELDS must include x,y,z")
    points = np.column_stack(
        (
            np.asarray(data["x"], dtype=np.float64),
            np.asarray(data["y"], dtype=np.float64),
            np.asarray(data["z"], dtype=np.float64),
        )
    )
    intensity = None
    if "intensity" in fields:
        intensity = np.asarray(data["intensity"], dtype=np.float64)
    viewpoint = np.asarray(metadata["viewpoint"], dtype=np.float64)
    default_view = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    vp = viewpoint if not np.allclose(viewpoint, default_view) else None
    return PcdScan(points=points, intensity=intensity, viewpoint=vp)


def load_pcd_scan(path: Path) -> PcdScan:
    """Load a PCD file including optional intensity and VIEWPOINT sensor pose."""
    header_lines: list[str] = []
    payload = b""
    data_kind = ""

    with path.open("rb") as f:
        while True:
            raw = f.readline()
            if not raw:
                break
            line = raw.decode("ascii", errors="strict").strip()
            header_lines.append(line)
            if line.lower().startswith("data "):
                data_kind = line.split()[1].lower()
                payload = f.read()
                break

    metadata, data_kind, _ = _parse_pcd_header(header_lines)
    if data_kind == "binary_compressed":
        raise ValueError("PCD DATA binary_compressed is not supported")

    fields = metadata["fields"]
    points = metadata["points"]
    idx = {k: fields.index(k) for k in ("x", "y", "z")}

    if data_kind == "ascii":
        point_lines = [
            ln for ln in payload.decode("utf-8").splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")
        ]
        if not point_lines:
            return PcdScan(points=np.zeros((0, 3), dtype=np.float64))
        table = np.loadtxt(io.StringIO("\n".join(point_lines)), dtype=np.float64)
        if table.ndim == 1:
            table = table[None, :]
        if table.shape[1] < len(fields):
            raise ValueError("PCD point format is shorter than expected")
        xyz = table[:, [idx["x"], idx["y"], idx["z"]]]
        intensity = table[:, fields.index("intensity")] if "intensity" in fields else None
        viewpoint = np.asarray(metadata["viewpoint"], dtype=np.float64)
        default_view = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        vp = viewpoint if not np.allclose(viewpoint, default_view) else None
        return PcdScan(points=xyz, intensity=intensity, viewpoint=vp)

    if data_kind != "binary":
        raise ValueError(f"unsupported PCD DATA type: {data_kind}")

    point_dtype = _pcd_structured_dtype(metadata)
    expected_size = point_dtype.itemsize * points
    if len(payload) < expected_size:
        raise ValueError("PCD binary payload is shorter than expected")
    structured = np.frombuffer(payload[:expected_size], dtype=point_dtype, count=points)
    return _structured_pcd_to_scan(structured, metadata)


def _load_pcd(path: Path) -> np.ndarray:
    return load_pcd_scan(path).points


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


def _spherical_pixels(
    points: np.ndarray,
    sensor_origin: np.ndarray,
    h_res_deg: float,
    v_res_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project points into a spherical range image around ``sensor_origin``.

    Returns ``(ranges, col, row, valid)`` where ``col``/``row`` are integer pixel
    indices (azimuth/elevation bins) and ``valid`` masks out points at the origin.
    """
    rel = np.asarray(points, dtype=np.float64) - sensor_origin
    x = rel[:, 0]
    y = rel[:, 1]
    z = rel[:, 2]
    ranges = np.sqrt(x * x + y * y + z * z)
    valid = ranges > 1e-9
    safe = np.where(valid, ranges, 1.0)
    azimuth = np.degrees(np.arctan2(y, x))  # (-180, 180]
    elevation = np.degrees(np.arcsin(np.clip(z / safe, -1.0, 1.0)))  # [-90, 90]
    col = np.floor((azimuth + 180.0) / h_res_deg).astype(np.int64)
    row = np.floor((elevation + 90.0) / v_res_deg).astype(np.int64)
    return ranges, col, row, valid


def remove_ghost_by_range_image(
    map_points: np.ndarray,
    query_points: np.ndarray,
    sensor_origin: Sequence[float] = (0.0, 0.0, 0.0),
    *,
    h_res_deg: float = DEFAULT_RANGE_H_RES_DEG,
    v_res_deg: float = DEFAULT_RANGE_V_RES_DEG,
    range_margin: float = DEFAULT_RANGE_MARGIN,
    min_query_points_per_pixel: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove ghost (dynamic) points from an accumulated map by visibility.

    This is the numpy "remove" step of range-image dynamic removal (Removert-style).
    Both ``map_points`` and ``query_points`` must be in the same frame. The query scan
    is the live sweep observed from ``sensor_origin``; a map point is considered a ghost
    when the query beam along its bearing passed *through* it and hit something farther,
    meaning that space is now free and the map point must have moved.

    A map point is removed when ``query_range - map_range > range_margin`` for its pixel.
    Pixels with no query return (unknown free space) are kept (conservative).

    Returns ``(kept_points, keep_mask)`` over ``map_points`` (same contract as
    :func:`remove_points_in_boxes`).
    """
    map_points = np.asarray(map_points, dtype=np.float64)
    if map_points.size == 0 or len(map_points) == 0:
        return map_points, np.ones(0, dtype=bool)

    query_points = np.asarray(query_points, dtype=np.float64)
    if query_points.size == 0 or len(query_points) == 0:
        # Nothing to test visibility against: keep everything.
        return map_points, np.ones(map_points.shape[0], dtype=bool)

    origin = np.asarray(sensor_origin, dtype=np.float64)
    if origin.shape != (3,):
        raise ValueError("sensor_origin must have 3 elements")

    n_cols = int(np.ceil(360.0 / h_res_deg))
    n_rows = int(np.ceil(180.0 / v_res_deg))

    q_ranges, q_col, q_row, q_valid = _spherical_pixels(query_points, origin, h_res_deg, v_res_deg)
    m_ranges, m_col, m_row, m_valid = _spherical_pixels(map_points, origin, h_res_deg, v_res_deg)

    # Build the query range image: nearest live return per pixel + a hit counter.
    q_col = np.clip(q_col, 0, n_cols - 1)
    q_row = np.clip(q_row, 0, n_rows - 1)
    flat_q = q_row * n_cols + q_col
    flat_q = flat_q[q_valid]

    nearest = np.full(n_rows * n_cols, np.inf, dtype=np.float64)
    np.minimum.at(nearest, flat_q, q_ranges[q_valid])
    counts = np.zeros(n_rows * n_cols, dtype=np.int64)
    np.add.at(counts, flat_q, 1)

    # Gather the query range at each map point's pixel.
    m_col_c = np.clip(m_col, 0, n_cols - 1)
    m_row_c = np.clip(m_row, 0, n_rows - 1)
    flat_m = m_row_c * n_cols + m_col_c
    pixel_query_range = nearest[flat_m]
    pixel_count = counts[flat_m]

    # Ghost when the live beam reached well past the map point in a sufficiently
    # observed pixel. Points at the origin (invalid) are kept.
    enough = pixel_count >= max(1, int(min_query_points_per_pixel))
    observed = np.isfinite(pixel_query_range) & enough
    ghost = observed & m_valid & ((pixel_query_range - m_ranges) > range_margin)
    keep_mask = ~ghost
    return map_points[keep_mask], keep_mask


def _visibility_votes(
    map_points: np.ndarray,
    query_points: np.ndarray,
    sensor_origin: np.ndarray,
    h_res_deg: float,
    v_res_deg: float,
    range_margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-map-point ``(seen_through, confirmed_surface)`` booleans for one query scan.

    ``seen_through``: the query beam reached well past the map point (free space).
    ``confirmed_surface``: the query saw a return at ~the map point's range (a real
    surface there). A point can be neither (unobserved pixel) but not both.
    """
    n_cols = int(np.ceil(360.0 / h_res_deg))
    n_rows = int(np.ceil(180.0 / v_res_deg))
    q_ranges, q_col, q_row, q_valid = _spherical_pixels(query_points, sensor_origin, h_res_deg, v_res_deg)
    flat_q = (np.clip(q_row, 0, n_rows - 1) * n_cols + np.clip(q_col, 0, n_cols - 1))[q_valid]
    nearest = np.full(n_rows * n_cols, np.inf, dtype=np.float64)
    np.minimum.at(nearest, flat_q, q_ranges[q_valid])

    m_ranges, m_col, m_row, m_valid = _spherical_pixels(map_points, sensor_origin, h_res_deg, v_res_deg)
    flat_m = np.clip(m_row, 0, n_rows - 1) * n_cols + np.clip(m_col, 0, n_cols - 1)
    pixel_q = nearest[flat_m]
    observed = np.isfinite(pixel_q) & m_valid
    seen_through = observed & ((pixel_q - m_ranges) > range_margin)
    confirmed = observed & (np.abs(pixel_q - m_ranges) < range_margin)
    return seen_through, confirmed


def clean_map_by_visibility(
    map_points: np.ndarray,
    scans: Sequence[tuple[np.ndarray, Sequence[float]]],
    *,
    h_res_deg: float = 1.0,
    v_res_deg: float = 1.0,
    range_margin: float = DEFAULT_RANGE_MARGIN,
    min_see_through: int = 2,
    max_surface_hits: int = 2,
    ground_z: float | None = None,
    resolutions: Sequence[float | tuple[float, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Clean an accumulated map of dynamic points using multi-scan visibility.

    This is the full range-image dynamic-removal pipeline (Removert/ERASOR family),
    pure numpy: each scan votes on every map point, and a point is removed only when
    the evidence agrees across scans.

    ``scans``: sequence of ``(points, sensor_origin)`` -- the pose-aligned sweeps that
    built the map (same frame as ``map_points``), each with the sensor position it was
    observed from.

    A map point is removed (dynamic) when it is **seen through** by at least
    ``min_see_through`` scans (free space along that beam) **and** confirmed as a real
    **surface** by at most ``max_surface_hits`` scans. The surface count is the *revert*
    guard from Removert: a repeatedly-observed static surface is kept even if a few
    scans see past it through occlusion gaps -- this is what stops the naive
    "remove-only" step from eroding static structure.

    ``ground_z``: if given, map points with ``z <= ground_z`` are protected (never
    removed); ground returns are sampled at shifting angles between scans and would
    otherwise generate spurious see-through votes.

    ``resolutions``: optional list of range-image resolutions for **multi-resolution
    consensus** (Removert-style). Each entry is a single degree value (square pixels) or
    an ``(h_deg, v_deg)`` pair. When given, a point is removed only if it is seen through
    at *every* resolution -- this filters resolution-specific noise and raises precision
    at the cost of a little recall, which helps on sparse sensors (e.g. nuScenes 32-beam,
    where a single fine image leaves too few points per pixel). The surface *revert* guard
    uses the finest resolution, where a real surface is best localized. When ``None``
    (default), a single ``(h_res_deg, v_res_deg)`` image is used (unchanged behaviour).

    Returns ``(kept_points, keep_mask)`` over ``map_points``.
    """
    map_points = np.asarray(map_points, dtype=np.float64)
    if map_points.size == 0 or len(map_points) == 0:
        return map_points, np.ones(0, dtype=bool)
    if not scans:
        return map_points, np.ones(map_points.shape[0], dtype=bool)

    if resolutions is None:
        res_list = [(float(h_res_deg), float(v_res_deg))]
    else:
        res_list = [((float(r), float(r)) if np.isscalar(r) else (float(r[0]), float(r[1])))
                    for r in resolutions]
        if not res_list:
            raise ValueError("resolutions must be a non-empty sequence")
    # Surface confirmation comes from the finest (smallest-cell) image, where a real
    # surface is most tightly localized; see-through must agree across all resolutions.
    finest = min(res_list, key=lambda hv: hv[0] * hv[1])

    n = map_points.shape[0]
    consensus_seen_through = np.ones(n, dtype=bool)
    surface_votes = np.zeros(n, dtype=np.int64)
    for h_deg, v_deg in res_list:
        see_through_votes = np.zeros(n, dtype=np.int64)
        sf_votes = np.zeros(n, dtype=np.int64)
        for pts, origin in scans:
            pts = np.asarray(pts, dtype=np.float64)
            if pts.size == 0:
                continue
            origin = np.asarray(origin, dtype=np.float64)
            st, sf = _visibility_votes(map_points, pts, origin, h_deg, v_deg, range_margin)
            see_through_votes += st.astype(np.int64)
            sf_votes += sf.astype(np.int64)
        consensus_seen_through &= see_through_votes >= max(1, int(min_see_through))
        if (h_deg, v_deg) == finest:
            surface_votes = sf_votes

    dynamic = consensus_seen_through & (surface_votes <= int(max_surface_hits))
    if ground_z is not None:
        dynamic &= map_points[:, 2] > ground_z
    keep_mask = ~dynamic
    return map_points[keep_mask], keep_mask


@dataclass
class RangeImageGhostFilter:
    """Streaming range-image ghost removal against a rolling local map.

    Keeps the last ``window_size`` scans as the reference map and, on each
    :meth:`filter` call, removes ghost points from the *incoming* scan by checking
    visibility against that rolling map, then appends the scan to the history.

    Assumes incoming scans share a frame (e.g. ego-motion compensated, or a static
    sensor). When that does not hold, pass per-scan ``sensor_origin`` and pre-aligned
    points. The first scan (empty history) is returned unchanged.
    """

    window_size: int = 5
    h_res_deg: float = DEFAULT_RANGE_H_RES_DEG
    v_res_deg: float = DEFAULT_RANGE_V_RES_DEG
    range_margin: float = DEFAULT_RANGE_MARGIN
    min_query_points_per_pixel: int = 1

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        self._history: deque[np.ndarray] = deque(maxlen=self.window_size)

    def filter(
        self,
        points: np.ndarray,
        sensor_origin: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=np.float64)
        if points.size == 0 or len(points) == 0:
            return points, np.ones(0, dtype=bool)

        if not self._history:
            self._history.append(points)
            return points, np.ones(points.shape[0], dtype=bool)

        ref_map = np.concatenate(list(self._history), axis=0)
        # Clean the incoming scan: it is the candidate set (map_points) and the rolling
        # history of past scans is the reference (query_points). A new point that the
        # past observed *past* (past range > new-point range) sits in previously-free
        # space -> it is a freshly-arrived dynamic point and is removed.
        kept, keep_mask = remove_ghost_by_range_image(
            points,
            ref_map,
            sensor_origin,
            h_res_deg=self.h_res_deg,
            v_res_deg=self.v_res_deg,
            range_margin=self.range_margin,
            min_query_points_per_pixel=self.min_query_points_per_pixel,
        )
        self._history.append(points)
        return kept, keep_mask


def _polar_bins(
    points: np.ndarray,
    sensor_origin: np.ndarray,
    n_rings: int,
    n_sectors: int,
    max_range: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign points to egocentric polar (ring x sector) bins around ``sensor_origin``.

    Returns ``(flat_bin, valid)`` where ``flat_bin = ring * n_sectors + sector`` and
    ``valid`` masks out the origin and anything beyond ``max_range``.
    """
    rel = np.asarray(points, dtype=np.float64) - sensor_origin
    x = rel[:, 0]
    y = rel[:, 1]
    radius = np.hypot(x, y)
    azimuth = np.arctan2(y, x)  # (-pi, pi]
    ring = np.floor(radius / max_range * n_rings).astype(np.int64)
    sector = np.clip(np.floor((azimuth + np.pi) / (2.0 * np.pi) * n_sectors).astype(np.int64), 0, n_sectors - 1)
    valid = (radius > 1e-9) & (ring >= 0) & (ring < n_rings)
    flat = ring * n_sectors + sector
    return flat, valid


def _ground_residual(pts: np.ndarray, seed_height: float) -> np.ndarray:
    """Height of each point above the local ground in one polar bin (R-GPF revert).

    Fits a ground plane ``z = a*x + b*y + c`` by least squares on the lowest points
    (within ``seed_height`` of the bin floor), refines once on the inliers, and returns
    ``z - plane(x, y)``. Falls back to "height above the lowest point" when the bin has
    too few points to fit a plane. Used to keep the ground while removing the object body
    that sits on top of it in a flagged column.
    """
    z = pts[:, 2]
    if pts.shape[0] < 3:
        return z - z.min()
    seed = z <= z.min() + seed_height
    if seed.sum() < 3:
        order = np.argsort(z)[:3]
        seed = np.zeros(z.shape[0], dtype=bool)
        seed[order] = True
    a = np.column_stack((pts[:, 0], pts[:, 1], np.ones(pts.shape[0])))
    coef, *_ = np.linalg.lstsq(a[seed], z[seed], rcond=None)
    resid = z - a @ coef
    # One refit on points close to the first plane, for robustness to the seed choice.
    inliers = np.abs(resid) <= seed_height
    if inliers.sum() >= 3:
        coef, *_ = np.linalg.lstsq(a[inliers], z[inliers], rcond=None)
        resid = z - a @ coef
    return resid


def _scan_ratio_dynamic(
    map_points: np.ndarray,
    query_points: np.ndarray,
    sensor_origin: np.ndarray,
    n_rings: int,
    n_sectors: int,
    max_range: float,
    scan_ratio_threshold: float,
    min_map_height: float,
    ground_margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-scan ``(dynamic, observed)`` masks over map points (ERASOR R-POD).

    A polar bin is "of interest" when the map has real vertical structure there
    (``map_height > min_map_height``) yet the live query bin is much flatter
    (``query_height / map_height < scan_ratio_threshold``) -- i.e. a tall thing in the
    map is gone in the current sweep. Inside such bins, points sitting above the local
    ground (residual ``> ground_margin``) are flagged dynamic; the ground is reverted.

    ``observed`` marks map points whose polar column is revisited by this scan
    (in-range and the query bin is non-empty) -- the points this sweep could have
    voted on at all. Used to normalize votes in :func:`clean_map_by_scan_ratio`.
    """
    n = map_points.shape[0]
    dynamic = np.zeros(n, dtype=bool)
    n_bins = n_rings * n_sectors

    m_flat, m_valid = _polar_bins(map_points, sensor_origin, n_rings, n_sectors, max_range)
    q_flat, q_valid = _polar_bins(query_points, sensor_origin, n_rings, n_sectors, max_range)
    mz = map_points[:, 2]
    qz = query_points[:, 2]

    def spread(flat: np.ndarray, valid: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hi = np.full(n_bins, -np.inf)
        lo = np.full(n_bins, np.inf)
        np.maximum.at(hi, flat[valid], z[valid])
        np.minimum.at(lo, flat[valid], z[valid])
        cnt = np.zeros(n_bins, dtype=np.int64)
        np.add.at(cnt, flat[valid], 1)
        height = np.where(cnt > 0, hi - lo, 0.0)
        return height, cnt

    map_h, map_cnt = spread(m_flat, m_valid, mz)
    q_h, q_cnt = spread(q_flat, q_valid, qz)

    observed = np.zeros(n, dtype=bool)
    observed[m_valid] = q_cnt[m_flat[m_valid]] > 0

    ratio = q_h / np.maximum(map_h, 1e-9)
    bin_interest = (map_cnt > 0) & (q_cnt > 0) & (map_h > min_map_height) & (ratio < scan_ratio_threshold)
    interest = np.nonzero(bin_interest)[0]
    if interest.size == 0:
        return dynamic, observed

    # Group valid map points by bin (one sort) so each flagged bin is a cheap slice.
    valid_idx = np.nonzero(m_valid)[0]
    bins_v = m_flat[valid_idx]
    order = np.argsort(bins_v, kind="stable")
    sorted_bins = bins_v[order]
    sorted_idx = valid_idx[order]
    starts = np.searchsorted(sorted_bins, interest, side="left")
    ends = np.searchsorted(sorted_bins, interest, side="right")

    seed_height = max(ground_margin * 2.0, 0.3)
    for b, s, e in zip(interest, starts, ends):
        if e <= s:
            continue
        idx = sorted_idx[s:e]
        resid = _ground_residual(map_points[idx], seed_height)
        dynamic[idx[resid > ground_margin]] = True
    return dynamic, observed


def remove_dynamic_by_scan_ratio(
    map_points: np.ndarray,
    query_points: np.ndarray,
    sensor_origin: Sequence[float] = (0.0, 0.0, 0.0),
    *,
    n_rings: int = DEFAULT_SR_RINGS,
    n_sectors: int = DEFAULT_SR_SECTORS,
    max_range: float = DEFAULT_SR_MAX_RANGE,
    scan_ratio_threshold: float = DEFAULT_SR_RATIO,
    min_map_height: float = DEFAULT_SR_MIN_MAP_HEIGHT,
    ground_margin: float = DEFAULT_SR_GROUND_MARGIN,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove dynamic points from a map by the ERASOR pseudo-occupancy scan-ratio test.

    Pure-numpy implementation of ERASOR's R-POD + scan-ratio + region-wise ground revert.
    This is a *different signal* from the range-image visibility family: instead of
    line-of-sight occlusion, it compares the **vertical occupancy** of each egocentric
    polar column between the accumulated ``map_points`` and a live ``query_points`` sweep
    (both in the same frame, observed from ``sensor_origin``). A column where the map is
    tall but the current scan is flat held a dynamic object whose trace must be removed;
    the ground underneath is reverted via a per-column least-squares plane fit.

    This catches dynamic traces that are never occluded (so visibility misses them) but is
    blind to objects floating off the ground and assumes dynamics rest on a visible ground.

    Returns ``(kept_points, keep_mask)`` over ``map_points`` (same contract as
    :func:`remove_points_in_boxes`).
    """
    map_points = np.asarray(map_points, dtype=np.float64)
    if map_points.size == 0 or len(map_points) == 0:
        return map_points, np.ones(0, dtype=bool)
    query_points = np.asarray(query_points, dtype=np.float64)
    if query_points.size == 0 or len(query_points) == 0:
        return map_points, np.ones(map_points.shape[0], dtype=bool)
    origin = np.asarray(sensor_origin, dtype=np.float64)
    if origin.shape != (3,):
        raise ValueError("sensor_origin must have 3 elements")

    dynamic, _ = _scan_ratio_dynamic(
        map_points, query_points, origin, int(n_rings), int(n_sectors), float(max_range),
        float(scan_ratio_threshold), float(min_map_height), float(ground_margin),
    )
    keep_mask = ~dynamic
    return map_points[keep_mask], keep_mask


def clean_map_by_scan_ratio(
    map_points: np.ndarray,
    scans: Sequence[tuple[np.ndarray, Sequence[float]]],
    *,
    n_rings: int = DEFAULT_SR_RINGS,
    n_sectors: int = DEFAULT_SR_SECTORS,
    max_range: float = DEFAULT_SR_MAX_RANGE,
    scan_ratio_threshold: float = DEFAULT_SR_RATIO,
    min_map_height: float = DEFAULT_SR_MIN_MAP_HEIGHT,
    ground_margin: float = DEFAULT_SR_GROUND_MARGIN,
    min_votes: int | None = None,
    votes_fraction: float = DEFAULT_SR_VOTES_FRACTION,
    votes_floor: int = DEFAULT_SR_VOTES_FLOOR,
) -> tuple[np.ndarray, np.ndarray]:
    """Clean an accumulated map with the scan-ratio test, voting across multiple scans.

    Runs :func:`remove_dynamic_by_scan_ratio` for each ``(points, sensor_origin)`` sweep
    and removes a map point only when enough scans flag it dynamic. Voting suppresses
    one-off false positives from occluded or sparsely-sampled sweeps (the main weakness
    of the per-column ratio test): a true dynamic trace is flagged by most sweeps that
    revisit its column (the object is gone), while a static surface only collects
    scattered votes.

    With ``min_votes=None`` (default) the vote threshold is normalized per point: a point
    is dynamic when ``votes >= max(votes_floor, ceil(votes_fraction * observed))``
    (``votes_floor`` is clamped to the number of scans), where
    ``observed`` counts the scans that actually revisited that point's polar column. The
    default ``votes_fraction=0.5`` is a majority rule over revisits: it protects static
    points seen only a handful of times (a fixed global threshold either over-deletes
    them or under-deletes well-observed traces). On DynamicMap_Benchmark Semantic-KITTI
    seq 00/05 this reaches SA 98.0/96.0 and DA 92.8/97.9 (AA 95.4/96.9) versus SA ~48%
    for a fixed ``min_votes=2``. Pass an integer ``min_votes`` to use a fixed absolute
    threshold instead. Mirrors :func:`clean_map_by_visibility`'s multi-scan shape.

    Returns ``(kept_points, keep_mask)`` over ``map_points``.
    """
    map_points = np.asarray(map_points, dtype=np.float64)
    if map_points.size == 0 or len(map_points) == 0:
        return map_points, np.ones(0, dtype=bool)
    if not scans:
        return map_points, np.ones(map_points.shape[0], dtype=bool)

    votes = np.zeros(map_points.shape[0], dtype=np.int64)
    observed = np.zeros(map_points.shape[0], dtype=np.int64)
    for pts, origin in scans:
        pts = np.asarray(pts, dtype=np.float64)
        if pts.size == 0:
            continue
        origin = np.asarray(origin, dtype=np.float64)
        dyn, obs = _scan_ratio_dynamic(
            map_points, pts, origin, int(n_rings), int(n_sectors), float(max_range),
            float(scan_ratio_threshold), float(min_map_height), float(ground_margin),
        )
        votes += dyn.astype(np.int64)
        observed += obs.astype(np.int64)
    if min_votes is None:
        floor = max(1, min(int(votes_floor), len(scans)))
        threshold = np.maximum(
            floor,
            np.ceil(float(votes_fraction) * observed).astype(np.int64),
        )
        dynamic = votes >= threshold
    else:
        dynamic = votes >= max(1, int(min_votes))
    keep_mask = ~dynamic
    return map_points[keep_mask], keep_mask


def _in_sorted(values: np.ndarray, sorted_arr: np.ndarray) -> np.ndarray:
    """Membership test of ``values`` against a sorted unique array."""
    if sorted_arr.size == 0:
        return np.zeros(values.shape, dtype=bool)
    idx = np.clip(np.searchsorted(sorted_arr, values), 0, sorted_arr.size - 1)
    return sorted_arr[idx] == values


def _voxel_grid(points: np.ndarray, voxel: float, pad: int = 2) -> tuple[np.ndarray, np.ndarray]:
    ijk = np.floor(points / voxel).astype(np.int64)
    mins = ijk.min(axis=0) - pad
    dims = ijk.max(axis=0) - mins + 1 + pad
    return mins, dims


def _voxel_keys(
    points: np.ndarray, mins: np.ndarray, dims: np.ndarray, voxel: float
) -> tuple[np.ndarray, np.ndarray]:
    ijk = np.floor(points / voxel).astype(np.int64) - mins
    ok = np.all((ijk >= 0) & (ijk < dims), axis=1)
    keys = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
    return keys, ok


def _ground_min_grid(points: np.ndarray, cell: float = 0.5) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    ij = np.floor(points[:, :2] / cell).astype(np.int64)
    mins = ij.min(axis=0)
    dims = ij.max(axis=0) - mins + 1
    flat = (ij[:, 0] - mins[0]) * dims[1] + (ij[:, 1] - mins[1])
    gz = np.full(int(dims[0] * dims[1]), np.inf)
    np.minimum.at(gz, flat, points[:, 2])
    return mins, dims, cell, gz


def _ground_z_at(xy: np.ndarray, grid: tuple[np.ndarray, np.ndarray, float, np.ndarray]) -> np.ndarray:
    mins, dims, cell, gz = grid
    ij = np.floor(xy / cell).astype(np.int64) - mins
    ij[:, 0] = np.clip(ij[:, 0], 0, dims[0] - 1)
    ij[:, 1] = np.clip(ij[:, 1], 0, dims[1] - 1)
    return gz[ij[:, 0] * dims[1] + ij[:, 1]]


_NB26 = np.array(
    [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)],
    dtype=np.int64,
)


def _carve_free_scan(
    map_keys: np.ndarray,
    pts: np.ndarray,
    origin: np.ndarray,
    grid: tuple[np.ndarray, np.ndarray, float],
    ground: tuple[np.ndarray, np.ndarray, float, np.ndarray],
    *,
    step: float,
    carve_margin: float,
    ground_margin: float,
    min_range: float,
    max_range: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Plain free-space carving for one scan: sample along rays, hit-precedence.

    Returns boolean ``(freed, observed)`` over the map points: ``freed`` marks
    points whose voxel was traversed by this scan's rays without being hit by
    it; ``observed`` marks points whose voxel was either traversed or hit.
    """
    mins, dims, voxel = grid
    vec = pts - origin
    r = np.linalg.norm(vec, axis=1)
    keep = r > min_range
    u = vec[keep] / r[keep, None]
    r = np.minimum(r[keep], max_range)
    endpts = pts[keep]

    ts = np.arange(min_range, max_range, step)
    free_chunks = []
    chunk = 20000
    for s in range(0, len(r), chunk):
        uc = u[s:s + chunk]
        rc = r[s:s + chunk]
        tmask = ts[None, :] < (rc[:, None] - carve_margin)
        if not tmask.any():
            continue
        n_t = int(tmask.sum(axis=1).max())
        samp = origin[None, None, :] + uc[:, None, :] * ts[None, :n_t, None]
        samp = samp[tmask[:, :n_t]]
        gz = _ground_z_at(samp[:, :2], ground)
        samp = samp[samp[:, 2] > gz + ground_margin]
        if not len(samp):
            continue
        k, ok = _voxel_keys(samp, mins, dims, voxel)
        free_chunks.append(np.unique(k[ok]))
    free = np.unique(np.concatenate(free_chunks)) if free_chunks else np.empty(0, np.int64)
    hit_keys, hok = _voxel_keys(endpts, mins, dims, voxel)
    hit = np.unique(hit_keys[hok])
    free = np.setdiff1d(free, hit, assume_unique=True)
    freed = _in_sorted(map_keys, free)
    observed = freed | _in_sorted(map_keys, hit)
    return freed, observed


def _carve_void_scan(
    map_keys: np.ndarray,
    pts: np.ndarray,
    origin: np.ndarray,
    grid: tuple[np.ndarray, np.ndarray, float],
    *,
    step: float,
    hit_inflation: float,
    min_range: float,
    max_range: float,
) -> np.ndarray:
    """Eroded void carving for one scan (DUFOMap-style confirmation).

    The last ``hit_inflation`` meters of every ray count as hit (sensor-noise
    guard, DUFOMap's d_s); miss rays stop on entering this scan's hit set; a
    miss voxel becomes a confirmed void only when all 26 neighbors were
    observed by this scan (DUFOMap's d_p erosion). Returns a boolean ``voided``
    mask over the map points.
    """
    mins, dims, voxel = grid
    vec = pts - origin
    r = np.linalg.norm(vec, axis=1)
    keep = (r > min_range) & (r < max_range)
    u = vec[keep] / r[keep, None]
    r = r[keep]
    endpts = pts[keep]

    hit_keys, hok = _voxel_keys(endpts, mins, dims, voxel)
    hit_list = [hit_keys[hok]]
    n_inf = max(1, int(np.ceil(hit_inflation / step)))
    for j in range(1, n_inf + 1):
        t = r - j * step
        good = t > min_range
        smp = origin[None, :] + u[good] * t[good, None]
        k, ok2 = _voxel_keys(smp, mins, dims, voxel)
        hit_list.append(k[ok2])
    hit = np.unique(np.concatenate(hit_list))

    ts = np.arange(min_range + step, max_range, step)
    miss_list, ext_list = [], []
    chunk = 8000
    n_ext = 2  # d_p + 1 voxels of observed-only ray extension past the stop
    for s in range(0, len(r), chunk):
        uc = u[s:s + chunk]
        rc = r[s:s + chunk]
        lim = rc - hit_inflation
        if not len(rc):
            continue
        n_t = int(min(len(ts), np.ceil((lim.max() - ts[0]) / step) + n_ext + 1))
        if n_t <= 0:
            continue
        tg = ts[:n_t]
        samp = origin[None, None, :] + uc[:, None, :] * tg[None, :, None]
        flat = samp.reshape(-1, 3)
        k, ok2 = _voxel_keys(flat, mins, dims, voxel)
        k = k.reshape(len(rc), n_t)
        ok2 = ok2.reshape(len(rc), n_t)
        within = tg[None, :] < lim[:, None]
        blocked = _in_sorted(k.ravel(), hit).reshape(len(rc), n_t) & within
        first_hit = np.where(blocked.any(axis=1), blocked.argmax(axis=1), n_t)
        col = np.arange(n_t)[None, :]
        miss_mask = within & ok2 & (col < first_hit[:, None])
        stop = np.minimum(
            np.where(blocked.any(axis=1), first_hit,
                     np.ceil((lim - ts[0]) / step).astype(np.int64)),
            n_t,
        )
        ext_mask = ok2 & (col >= stop[:, None]) & (col < (stop + n_ext)[:, None])
        miss_list.append(np.unique(k[miss_mask]))
        ext_list.append(np.unique(k[ext_mask]))
    miss = np.unique(np.concatenate(miss_list)) if miss_list else np.empty(0, np.int64)
    miss = np.setdiff1d(miss, hit, assume_unique=True)
    ext = np.unique(np.concatenate(ext_list)) if ext_list else np.empty(0, np.int64)
    observed = np.union1d(np.union1d(miss, hit), ext)

    noff = _NB26[:, 0] * int(dims[1] * dims[2]) + _NB26[:, 1] * int(dims[2]) + _NB26[:, 2]
    confirmed = np.ones(miss.shape, dtype=bool)
    for off in noff:
        confirmed &= _in_sorted(miss + off, observed)
        if not confirmed.any():
            break
    return _in_sorted(map_keys, miss[confirmed])


_FUSION_STATE: dict[str, Any] | None = None


def _fusion_accumulate(
    map_points: np.ndarray,
    scans: Sequence[tuple[np.ndarray, Sequence[float]]],
    indices: Sequence[int],
    params: dict[str, Any],
) -> tuple[np.ndarray, ...]:
    n = map_points.shape[0]
    free_grid = params["free_grid"]
    void_grid = params["void_grid"]
    ground = params["ground"]
    free_keys = params["free_keys"]
    void_keys = params["void_keys"]
    sr_votes = np.zeros(n, dtype=np.int32)
    sr_obs = np.zeros(n, dtype=np.int32)
    free_votes = np.zeros(n, dtype=np.int32)
    free_obs = np.zeros(n, dtype=np.int32)
    void_votes = np.zeros(n, dtype=np.int32)
    for i in indices:
        pts, origin = scans[i]
        pts = np.asarray(pts, dtype=np.float64)
        if pts.size == 0:
            continue
        origin = np.asarray(origin, dtype=np.float64)
        dyn, obs = _scan_ratio_dynamic(
            map_points, pts, origin,
            params["n_rings"], params["n_sectors"], params["sr_max_range"],
            params["scan_ratio_threshold"], params["min_map_height"],
            params["sr_ground_margin"],
        )
        sr_votes += dyn.astype(np.int32)
        sr_obs += obs.astype(np.int32)
        freed, fobs = _carve_free_scan(
            free_keys, pts, origin, free_grid, ground,
            step=params["free_step"], carve_margin=params["free_carve_margin"],
            ground_margin=params["free_ground_margin"],
            min_range=params["min_range"], max_range=params["max_range"],
        )
        free_votes += freed.astype(np.int32)
        free_obs += fobs.astype(np.int32)
        voided = _carve_void_scan(
            void_keys, pts, origin, void_grid,
            step=params["void_step"], hit_inflation=params["void_hit_inflation"],
            min_range=params["min_range"], max_range=params["max_range"],
        )
        void_votes += voided.astype(np.int32)
    return sr_votes, sr_obs, free_votes, free_obs, void_votes


def _fusion_worker(indices: Sequence[int]) -> tuple[np.ndarray, ...]:
    assert _FUSION_STATE is not None
    return _fusion_accumulate(
        _FUSION_STATE["map_points"], _FUSION_STATE["scans"], indices, _FUSION_STATE["params"],
    )


def clean_map_by_fusion(
    map_points: np.ndarray,
    scans: Sequence[tuple[np.ndarray, Sequence[float]]],
    *,
    min_range: float = DEFAULT_FUSION_MIN_RANGE,
    max_range: float = DEFAULT_FUSION_MAX_RANGE,
    free_voxel: float = DEFAULT_FREE_VOXEL,
    free_step: float = DEFAULT_FREE_STEP,
    free_carve_margin: float = DEFAULT_FREE_CARVE_MARGIN,
    free_ground_margin: float = DEFAULT_FREE_GROUND_MARGIN,
    free_votes_fraction: float = DEFAULT_FREE_VOTES_FRACTION,
    free_votes_floor: int = DEFAULT_FREE_VOTES_FLOOR,
    void_voxel: float = DEFAULT_VOID_VOXEL,
    void_step: float = DEFAULT_VOID_STEP,
    void_hit_inflation: float = DEFAULT_VOID_HIT_INFLATION,
    void_min_scans: int = DEFAULT_VOID_MIN_SCANS,
    sr_votes_fraction: float = DEFAULT_FUSION_SR_FRACTION,
    sr_votes_floor: int = DEFAULT_SR_VOTES_FLOOR,
    n_rings: int = DEFAULT_SR_RINGS,
    n_sectors: int = DEFAULT_SR_SECTORS,
    sr_max_range: float = DEFAULT_SR_MAX_RANGE,
    scan_ratio_threshold: float = DEFAULT_SR_RATIO,
    min_map_height: float = DEFAULT_SR_MIN_MAP_HEIGHT,
    sr_ground_margin: float = DEFAULT_SR_GROUND_MARGIN,
    workers: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Clean an accumulated map by OR-fusing three dynamic-evidence channels.

    Channels, each computed per scan against the accumulated map:

    1. *Free-space carving*: rays are sampled every ``free_step`` meters and
       stop ``free_carve_margin`` before their endpoint; voxels traversed but
       not hit by the same scan are freed (samples below the local ground
       level + ``free_ground_margin`` never carve). A point is dynamic when
       freed by ``free_votes_fraction`` of the scans that observed its voxel.
       High fractions capture transient traffic with near-perfect precision.
    2. *Eroded voids* (DUFOMap-style): finer carving where the last
       ``void_hit_inflation`` meters of every ray count as hit, miss rays stop
       at the scan's own hit set, and a miss is confirmed only when its full
       26-neighborhood was observed in the same scan. A point is dynamic after
       ``void_min_scans`` confirmed voids (an absolute count, so slow movers
       and late leavers are caught even when they occupied the spot for most
       of the sequence — a regime where fractional voting fails).
    3. *Scan-ratio votes* (see :func:`clean_map_by_scan_ratio`) thresholded at
       the stricter ``sr_votes_fraction`` — a high-precision polar-column
       signal that needs no ray geometry.

    The union of the three channels removes complementary error pools.
    Measured on DynamicMap_Benchmark Semantic-KITTI seq 00 / 05:
    SA 98.9 / 98.0, DA 98.3 / 98.1, AA 98.6 / 98.0 — matching DUFOMap (the
    strongest method on the public leaderboard, AA 98.6 / 96.3) on seq 00 and
    surpassing it on seq 05.

    ``workers > 1`` distributes scans over a fork-based process pool (Linux /
    macOS); on platforms without ``fork`` it falls back to sequential.
    Carving cost dominates: expect a few minutes per 100 scans of 64-beam
    data at the defaults with ``workers=6``.

    Returns ``(kept_points, keep_mask)`` over ``map_points``.
    """
    global _FUSION_STATE
    map_points = np.asarray(map_points, dtype=np.float64)
    if map_points.size == 0 or len(map_points) == 0:
        return map_points, np.ones(0, dtype=bool)
    if not scans:
        return map_points, np.ones(map_points.shape[0], dtype=bool)

    free_mins, free_dims = _voxel_grid(map_points, float(free_voxel))
    void_mins, void_dims = _voxel_grid(map_points, float(void_voxel))
    free_keys, _ = _voxel_keys(map_points, free_mins, free_dims, float(free_voxel))
    void_keys, _ = _voxel_keys(map_points, void_mins, void_dims, float(void_voxel))
    params: dict[str, Any] = {
        "min_range": float(min_range), "max_range": float(max_range),
        "free_grid": (free_mins, free_dims, float(free_voxel)),
        "void_grid": (void_mins, void_dims, float(void_voxel)),
        "ground": _ground_min_grid(map_points),
        "free_keys": free_keys, "void_keys": void_keys,
        "free_step": float(free_step), "free_carve_margin": float(free_carve_margin),
        "free_ground_margin": float(free_ground_margin),
        "void_step": float(void_step), "void_hit_inflation": float(void_hit_inflation),
        "n_rings": int(n_rings), "n_sectors": int(n_sectors),
        "sr_max_range": float(sr_max_range),
        "scan_ratio_threshold": float(scan_ratio_threshold),
        "min_map_height": float(min_map_height),
        "sr_ground_margin": float(sr_ground_margin),
    }

    indices = list(range(len(scans)))
    workers = max(1, int(workers))
    partials: list[tuple[np.ndarray, ...]]
    if workers > 1:
        import multiprocessing

        try:
            ctx = multiprocessing.get_context("fork")
        except ValueError:
            ctx = None
        if ctx is not None:
            _FUSION_STATE = {"map_points": map_points, "scans": scans, "params": params}
            try:
                splits = [list(c) for c in np.array_split(np.asarray(indices), workers) if len(c)]
                with ctx.Pool(len(splits)) as pool:
                    partials = pool.map(_fusion_worker, splits)
            finally:
                _FUSION_STATE = None
        else:
            _eprint("fusion: fork unavailable, running sequentially")
            partials = [_fusion_accumulate(map_points, scans, indices, params)]
    else:
        partials = [_fusion_accumulate(map_points, scans, indices, params)]

    n = map_points.shape[0]
    sr_votes = np.zeros(n, dtype=np.int64)
    sr_obs = np.zeros(n, dtype=np.int64)
    free_votes = np.zeros(n, dtype=np.int64)
    free_obs = np.zeros(n, dtype=np.int64)
    void_votes = np.zeros(n, dtype=np.int64)
    for sv, so, fv, fo, vv in partials:
        sr_votes += sv
        sr_obs += so
        free_votes += fv
        free_obs += fo
        void_votes += vv

    sr_floor = max(1, min(int(sr_votes_floor), len(scans)))
    sr_dyn = sr_votes >= np.maximum(
        sr_floor, np.ceil(float(sr_votes_fraction) * sr_obs).astype(np.int64)
    )
    free_floor = max(1, min(int(free_votes_floor), len(scans)))
    free_dyn = free_votes >= np.maximum(
        free_floor, np.ceil(float(free_votes_fraction) * free_obs).astype(np.int64)
    )
    void_dyn = void_votes >= max(1, min(int(void_min_scans), len(scans)))
    keep_mask = ~(sr_dyn | free_dyn | void_dyn)
    return map_points[keep_mask], keep_mask


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
    parser = argparse.ArgumentParser(description="Remove dynamic points from point clouds (box or range-image visibility).")
    parser.add_argument("--input-cloud", required=True, help="Input point cloud path (csv/txt/xyz/pcd/npy). For --algorithm range this is the query scan.")
    parser.add_argument("--input-objects", help="Detected object boxes JSON or CSV path (required for --algorithm box).")
    parser.add_argument("--output-cloud", required=True, help="Output point cloud path.")
    parser.add_argument("--algorithm", choices=["box", "range", "scan_ratio"], default="box", help="box: crop by detection boxes. range: range-image visibility removal of an accumulated map. scan_ratio: ERASOR-style per-column pseudo-occupancy removal of an accumulated map.")
    parser.add_argument("--input-map", help="Accumulated map point cloud to clean (required for --algorithm range/scan_ratio).")
    parser.add_argument("--sensor-origin", nargs=3, type=float, default=[0.0, 0.0, 0.0], metavar=("X", "Y", "Z"), help="Sensor origin of the query scan (meters), for --algorithm range/scan_ratio.")
    parser.add_argument("--range-margin", type=float, default=DEFAULT_RANGE_MARGIN, help="Free-space margin for range-image removal (meters).")
    parser.add_argument("--range-h-res", type=float, default=DEFAULT_RANGE_H_RES_DEG, help="Range-image azimuth resolution (degrees).")
    parser.add_argument("--range-v-res", type=float, default=DEFAULT_RANGE_V_RES_DEG, help="Range-image elevation resolution (degrees).")
    parser.add_argument("--scan-ratio-threshold", type=float, default=DEFAULT_SR_RATIO, help="scan_ratio: a column is dynamic when query/map height ratio is below this.")
    parser.add_argument("--scan-ratio-min-map-height", type=float, default=DEFAULT_SR_MIN_MAP_HEIGHT, help="scan_ratio: ignore columns whose map height spread is below this (meters).")
    parser.add_argument("--scan-ratio-ground-margin", type=float, default=DEFAULT_SR_GROUND_MARGIN, help="scan_ratio: keep points within this height of the per-column ground (meters).")
    parser.add_argument("--cloud-format", default="auto", choices=["auto", "csv", "pcd", "text", "npy", "bin", "feather"], help="Output/input point cloud format.")
    parser.add_argument("--objects-format", default="auto", choices=["auto", "json", "csv", "kitti", "av2"], help="Object file format.")
    parser.add_argument("--calib-path", default=None, help="KITTI calibration file path (required when --objects-format=kitti).")
    parser.add_argument("--timestamp-ns", type=int, default=None, help="Filter AV2 annotations by timestamp (nanoseconds).")
    parser.add_argument("--box-margin", nargs=3, type=float, default=list(DEFAULT_BOX_MARGIN), metavar=("X", "Y", "Z"), help="Safety margin around each box (meters).")
    parser.add_argument("--skip-invalid", action="store_true", help="Skip invalid object entries instead of stopping.")
    parser.add_argument("--min-size", type=float, default=0.01, help="Skip boxes smaller than this size in any axis.")
    parser.add_argument("--summary-json", help="Write filtering statistics as JSON to this path.")
    parser.add_argument("--quiet", action="store_true", help="Suppress stdout summary.")
    parser.add_argument("--version", action="version", version=f"dynamic-object-removal {__version__}")
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
    out_path = Path(args.output_cloud)

    if not cloud_path.exists():
        _eprint(f"input cloud not found: {cloud_path}")
        return 1

    if args.algorithm == "range":
        if not args.input_map:
            _eprint("algorithm=range requires --input-map (the accumulated map to clean)")
            return 1
        map_path = Path(args.input_map)
        if not map_path.exists():
            _eprint(f"input map not found: {map_path}")
            return 1
        map_points = load_points(map_path, fmt=args.cloud_format)
        query_points = load_points(cloud_path, fmt=args.cloud_format)
        filtered, keep_mask = remove_ghost_by_range_image(
            map_points,
            query_points,
            tuple(args.sensor_origin),
            h_res_deg=args.range_h_res,
            v_res_deg=args.range_v_res,
            range_margin=args.range_margin,
        )
        removed = map_points.shape[0] - filtered.shape[0]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_points(out_path, filtered, fmt=args.cloud_format)
        if not args.quiet:
            ratio = 0.0 if map_points.shape[0] == 0 else removed / map_points.shape[0]
            _eprint(f"algorithm: range (visibility)")
            _eprint(f"map: {map_points.shape[0]} points, query: {query_points.shape[0]} points")
            _eprint(f"removed: {removed} points ({ratio:.2%})")
            _eprint(f"output: {filtered.shape[0]} points -> {out_path}")
        if args.summary_json:
            payload = {
                "algorithm": "range",
                "total_points": int(map_points.shape[0]),
                "kept_points": int(filtered.shape[0]),
                "removed_points": int(removed),
                "removed_ratio": float(removed / map_points.shape[0]) if map_points.shape[0] else 0.0,
            }
            Path(args.summary_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0

    if args.algorithm == "scan_ratio":
        if not args.input_map:
            _eprint("algorithm=scan_ratio requires --input-map (the accumulated map to clean)")
            return 1
        map_path = Path(args.input_map)
        if not map_path.exists():
            _eprint(f"input map not found: {map_path}")
            return 1
        map_points = load_points(map_path, fmt=args.cloud_format)
        query_points = load_points(cloud_path, fmt=args.cloud_format)
        filtered, keep_mask = remove_dynamic_by_scan_ratio(
            map_points,
            query_points,
            tuple(args.sensor_origin),
            scan_ratio_threshold=args.scan_ratio_threshold,
            min_map_height=args.scan_ratio_min_map_height,
            ground_margin=args.scan_ratio_ground_margin,
        )
        removed = map_points.shape[0] - filtered.shape[0]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_points(out_path, filtered, fmt=args.cloud_format)
        if not args.quiet:
            ratio = 0.0 if map_points.shape[0] == 0 else removed / map_points.shape[0]
            _eprint(f"algorithm: scan_ratio (pseudo-occupancy)")
            _eprint(f"map: {map_points.shape[0]} points, query: {query_points.shape[0]} points")
            _eprint(f"removed: {removed} points ({ratio:.2%})")
            _eprint(f"output: {filtered.shape[0]} points -> {out_path}")
        if args.summary_json:
            payload = {
                "algorithm": "scan_ratio",
                "total_points": int(map_points.shape[0]),
                "kept_points": int(filtered.shape[0]),
                "removed_points": int(removed),
                "removed_ratio": float(removed / map_points.shape[0]) if map_points.shape[0] else 0.0,
            }
            Path(args.summary_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0

    # algorithm == "box"
    if not args.input_objects:
        _eprint("algorithm=box requires --input-objects")
        return 1
    obj_path = Path(args.input_objects)
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
