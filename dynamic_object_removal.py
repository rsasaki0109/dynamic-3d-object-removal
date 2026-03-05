#!/usr/bin/env python3
"""Dynamic object removal CLI.

This tool removes points that lie inside detected 3D bounding boxes.
It follows the same practical idea as the `dynamic_object_removal` ROS node:
crop each incoming point cloud by detected object boxes and keep only static points.
"""

from __future__ import annotations

import argparse
import csv
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


def load_boxes(path: Path, *, fmt: str, skip_invalid: bool) -> list[DetectionBox]:
    fmt = fmt.lower()
    if fmt == "auto":
        if path.suffix.lower() in {".json", ".jsn"}:
            fmt = "json"
        elif path.suffix.lower() in {".csv", ".tsv", ".txt"}:
            fmt = "csv"
        else:
            fmt = "json"
    if fmt == "json":
        return _load_boxes_from_json(path, skip_invalid=skip_invalid)
    if fmt == "csv":
        return _load_boxes_from_csv(path, skip_invalid=skip_invalid)
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


def _load_pcd_ascii(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="utf-8").splitlines()
    fields: list[str] = []
    data_start = 0
    for i, line in enumerate(lines):
        low = line.strip().lower()
        if low.startswith("fields "):
            fields = [x for x in low.split()[1:]]
        elif low.startswith("data "):
            if "ascii" not in low:
                raise ValueError("only ASCII PCD is supported currently")
            data_start = i + 1
            break
    if not fields:
        raise ValueError("PCD header missing FIELDS line")
    if not data_start or data_start >= len(lines):
        raise ValueError("PCD has no DATA ascii section")

    idx = {k: fields.index(k) for k in ("x", "y", "z") if k in fields}
    if len(idx) != 3:
        raise ValueError("PCD FIELDS must include x,y,z")
    point_lines = [ln for ln in lines[data_start:] if ln.strip() and not ln.lstrip().startswith("#")]
    if not point_lines:
        return np.zeros((0, 3), dtype=np.float64)
    data = np.loadtxt(point_lines, dtype=np.float64)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < len(fields):
        raise ValueError("PCD point format is shorter than expected")
    return data[:, [idx["x"], idx["y"], idx["z"]]]


def load_points(path: Path, *, fmt: str) -> np.ndarray:
    fmt = fmt.lower()
    if fmt == "auto":
        if path.suffix.lower() == ".npy":
            fmt = "npy"
        elif path.suffix.lower() == ".pcd":
            fmt = "pcd"
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
        return _load_pcd_ascii(path)
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


def remove_points_in_boxes(points: np.ndarray, boxes: Sequence[DetectionBox], margin: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
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
    parser.add_argument("--cloud-format", default="auto", choices=["auto", "csv", "pcd", "text", "npy"], help="Output/input point cloud format.")
    parser.add_argument("--objects-format", default="auto", choices=["auto", "json", "csv"], help="Object file format.")
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

    boxes = load_boxes(obj_path, fmt=args.objects_format, skip_invalid=args.skip_invalid)
    boxes = _filter_small_boxes(boxes, args.min_size)

    if not boxes:
        _eprint("no valid boxes. nothing will be removed.")

    points = load_points(cloud_path, fmt=args.cloud_format)
    filtered, keep_mask = remove_points_in_boxes(points, boxes, args.box_margin)

    removed = points.shape[0] - filtered.shape[0]
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
