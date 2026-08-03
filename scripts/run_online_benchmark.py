#!/usr/bin/env python3
"""Replay a pose-aligned LiDAR sequence once and benchmark streaming filters.

This benchmark is intentionally different from ``dynamic-object-removal-bench``:
the latter is a microbenchmark that repeats one cloud, while this script processes a
sequence exactly once in timestamp order. It measures online point-wise accuracy,
static preservation, warm-up/confirmation delay, filter latency, deadline misses,
fail-open frames, and sensitivity to pose noise.

Input is a JSON manifest. Paths are relative to the manifest unless absolute::

    {
      "sensor_profile": {"name": "VLP-16", "beams": 16, "rate_hz": 10,
                         "deskewed": true},
      "frames": [
        {"cloud": "frames/000.npy", "timestamp_sec": 0.0,
         "pose": {"translation": [0, 0, 0],
                  "quaternion_xyzw": [0, 0, 0, 1]},
         "point_labels": "labels/000.npy"}
      ]
    }

``point_labels`` is a per-point array where nonzero means currently moving. Instead,
``dynamic_boxes`` may contain the same box objects accepted by
``parse_boxes_payload``; boxes are interpreted in the sensor frame. Every cloud must
already be deskewed because one rigid pose is applied at its frame timestamp.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bench  # noqa: E402
import dynamic_object_removal as core  # noqa: E402
from scripts import online_mapping_backend as mapping_backend  # noqa: E402


@dataclass(frozen=True)
class Pose:
    rotation: np.ndarray
    translation: np.ndarray


def _rotation_from_quaternion_xyzw(value: Sequence[float]) -> np.ndarray:
    q = np.asarray(value, dtype=np.float64).reshape(-1)
    if q.size != 4:
        raise ValueError("quaternion_xyzw must have four values")
    x, y, z, w = map(float, q)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        raise ValueError("pose quaternion has zero norm")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _pose_from_payload(value: Any) -> Pose:
    if not isinstance(value, dict):
        raise ValueError("pose must be an object")
    translation = np.asarray(value.get("translation"), dtype=np.float64).reshape(-1)
    if translation.size != 3:
        raise ValueError("pose.translation must have three values")
    if "quaternion_xyzw" in value:
        rotation = _rotation_from_quaternion_xyzw(value["quaternion_xyzw"])
    elif "rotation" in value:
        rotation = np.asarray(value["rotation"], dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError("pose.rotation must have shape (3,3)")
    else:
        raise ValueError("pose requires quaternion_xyzw or rotation")
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ValueError("pose contains non-finite values")
    return Pose(rotation=rotation, translation=translation)


def _yaw_rotation(yaw_rad: float) -> np.ndarray:
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _perturb_pose(
    pose: Pose,
    *,
    translation_sigma: float,
    yaw_sigma_deg: float,
    rng: np.random.Generator,
) -> Pose:
    translation = pose.translation + rng.normal(0.0, translation_sigma, size=3)
    yaw = math.radians(float(rng.normal(0.0, yaw_sigma_deg)))
    return Pose(rotation=_yaw_rotation(yaw) @ pose.rotation, translation=translation)


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_gt_mask(frame: dict[str, Any], points: np.ndarray, root: Path) -> np.ndarray:
    if "point_labels" in frame:
        path = _resolve_path(root, frame["point_labels"])
        if path.suffix.lower() == ".npy":
            labels = np.load(path)
        else:
            labels = np.loadtxt(path)
        labels = np.asarray(labels).reshape(-1)
        if len(labels) != len(points):
            raise ValueError(
                f"point label count {len(labels)} does not match cloud count {len(points)}: {path}"
            )
        return labels != 0
    if "dynamic_boxes" in frame:
        boxes = core.parse_boxes_payload(frame["dynamic_boxes"], skip_invalid=False)
        return bench.dynamic_gt_mask(points, boxes, dynamic_labels=None)
    raise ValueError("each frame requires point_labels or dynamic_boxes")


def _new_filter(
    args: argparse.Namespace,
    *,
    algorithm: str | None = None,
    temporal_visibility: bool | None = None,
    temporal_h_res: float | None = None,
    temporal_v_res: float | None = None,
) -> Any:
    selected_algorithm = algorithm or args.algorithm
    if selected_algorithm == "temporal":
        use_visibility = (
            bool(getattr(args, "temporal_visibility", False))
            if temporal_visibility is None
            else bool(temporal_visibility)
        )
        return core.TemporalConsistencyFilter(
            voxel_size=args.voxel_size,
            window_size=args.temporal_window,
            min_hits=args.temporal_min_hits,
            visibility=use_visibility,
            visibility_h_res_deg=(
                float(getattr(args, "temporal_visibility_h_res", core.DEFAULT_RANGE_H_RES_DEG))
                if temporal_h_res is None
                else float(temporal_h_res)
            ),
            visibility_v_res_deg=(
                float(getattr(args, "temporal_visibility_v_res", core.DEFAULT_RANGE_V_RES_DEG))
                if temporal_v_res is None
                else float(temporal_v_res)
            ),
            visibility_margin=float(
                getattr(args, "temporal_visibility_margin", core.DEFAULT_RANGE_MARGIN)
            ),
            visibility_fraction=float(
                getattr(args, "temporal_visibility_fraction", 0.30)
            ),
            visibility_min_hits=getattr(args, "temporal_visibility_min_hits", 1),
        )
    return core.RangeImageGhostFilter(
        window_size=args.range_window,
        h_res_deg=args.range_h_res,
        v_res_deg=args.range_v_res,
        range_margin=args.range_margin,
    )


def _latency_summary(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "max_ms": float(arr.max()),
    }


def _scenario_name(translation_sigma: float, yaw_sigma_deg: float) -> str:
    if translation_sigma == 0.0 and yaw_sigma_deg == 0.0:
        return "baseline"
    if translation_sigma:
        return f"translation_sigma_{translation_sigma:g}m"
    return f"yaw_sigma_{yaw_sigma_deg:g}deg"


def replay_scenario(
    manifest: dict[str, Any],
    manifest_root: Path,
    args: argparse.Namespace,
    *,
    translation_sigma: float = 0.0,
    yaw_sigma_deg: float = 0.0,
) -> dict[str, Any]:
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("manifest.frames must be a non-empty list")
    filterer = _new_filter(args)
    rng = np.random.default_rng(args.seed)
    removed_chunks: list[np.ndarray] = []
    gt_chunks: list[np.ndarray] = []
    latencies_ms: list[float] = []
    per_frame: list[dict[str, Any]] = []
    fail_open_frames = 0
    first_timestamp: float | None = None
    confirmation_frame: int | None = None
    confirmation_timestamp: float | None = None
    warmup_frames = args.temporal_min_hits - 1 if args.algorithm == "temporal" else 1

    for index, frame in enumerate(frames):
        if not isinstance(frame, dict) or "cloud" not in frame:
            raise ValueError(f"invalid frame {index}: cloud is required")
        cloud_path = _resolve_path(manifest_root, frame["cloud"])
        points = core.load_points(cloud_path, fmt="auto")
        gt_dynamic = _load_gt_mask(frame, points, manifest_root)
        timestamp = float(frame.get("timestamp_sec", index))
        if first_timestamp is None:
            first_timestamp = timestamp

        start = time.perf_counter()
        pose_payload = frame.get("pose")
        if pose_payload is None:
            if args.missing_pose == "error":
                raise ValueError(f"frame {index} has no pose")
            keep_mask = np.ones(len(points), dtype=bool)
            fail_open_frames += 1
        else:
            pose = _perturb_pose(
                _pose_from_payload(pose_payload),
                translation_sigma=translation_sigma,
                yaw_sigma_deg=yaw_sigma_deg,
                rng=rng,
            )
            fixed_points = points @ pose.rotation.T + pose.translation
            if args.algorithm == "temporal":
                _, keep_mask = filterer.filter(fixed_points)
            else:
                _, keep_mask = filterer.filter(fixed_points, pose.translation)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(elapsed_ms)
        removed = ~keep_mask
        removed_chunks.append(removed)
        gt_chunks.append(gt_dynamic)
        metrics = bench.compute_accuracy_metrics(removed, gt_dynamic)
        static_keep = metrics["static_preservation"]
        if (
            confirmation_frame is None
            and index >= warmup_frames
            and static_keep >= args.confirmation_static_keep
        ):
            confirmation_frame = index
            confirmation_timestamp = timestamp
        per_frame.append(
            {
                "index": index,
                "timestamp_sec": timestamp,
                "points": int(len(points)),
                "gt_dynamic_points": int(np.count_nonzero(gt_dynamic)),
                "removed_points": int(np.count_nonzero(removed)),
                "fail_open": pose_payload is None,
                "filter_latency_ms": float(elapsed_ms),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "iou": metrics["iou"],
                "static_preservation": static_keep,
                "true_positive": metrics["true_positive"],
                "false_positive": metrics["false_positive"],
                "false_negative": metrics["false_negative"],
                "true_negative": metrics["true_negative"],
            }
        )

    removed_all = np.concatenate(removed_chunks)
    gt_all = np.concatenate(gt_chunks)
    metrics = bench.compute_accuracy_metrics(removed_all, gt_all)
    profile = manifest.get("sensor_profile") if isinstance(manifest.get("sensor_profile"), dict) else {}
    rate_hz = float(args.rate_hz or profile.get("rate_hz") or 0.0)
    period_ms = 1000.0 / rate_hz if rate_hz > 0.0 else None
    deadline_misses = (
        int(np.count_nonzero(np.asarray(latencies_ms) > period_ms)) if period_ms is not None else None
    )
    return {
        "name": _scenario_name(translation_sigma, yaw_sigma_deg),
        "pose_noise": {
            "translation_sigma_m": float(translation_sigma),
            "yaw_sigma_deg": float(yaw_sigma_deg),
            "seed": int(args.seed),
        },
        "frames": len(frames),
        "points": int(len(removed_all)),
        "metrics": metrics,
        "warmup_frames": int(warmup_frames),
        "confirmation_static_keep_threshold": float(args.confirmation_static_keep),
        "time_to_confirm_frame": confirmation_frame,
        "time_to_confirm_sec": (
            float(confirmation_timestamp - first_timestamp)
            if confirmation_timestamp is not None and first_timestamp is not None
            else None
        ),
        "filter_latency": _latency_summary(latencies_ms),
        "rate_hz": rate_hz or None,
        "period_ms": period_ms,
        "deadline_misses": deadline_misses,
        "dropped_frames": 0,
        "fail_open_frames": fail_open_frames,
        "per_frame": per_frame,
    }


class _GrowingMap:
    """Small append-only map buffer so backend indices stay stable."""

    def __init__(self) -> None:
        self.points = np.empty((0, 3), dtype=np.float64)
        self.gt = np.empty(0, dtype=bool)
        self.active = np.empty(0, dtype=bool)
        self.size = 0

    def append(self, points: np.ndarray, gt_dynamic: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        gt_dynamic = np.asarray(gt_dynamic, dtype=bool).reshape(-1)
        if len(points) != len(gt_dynamic):
            raise ValueError("map points and labels must have equal length")
        if not len(points):
            return np.empty(0, dtype=np.int64)
        required = self.size + len(points)
        if required > len(self.points):
            capacity = max(required, max(1024, len(self.points) * 2))
            new_points = np.empty((capacity, 3), dtype=np.float64)
            new_gt = np.empty(capacity, dtype=bool)
            new_active = np.empty(capacity, dtype=bool)
            if self.size:
                new_points[: self.size] = self.points[: self.size]
                new_gt[: self.size] = self.gt[: self.size]
                new_active[: self.size] = self.active[: self.size]
            self.points, self.gt, self.active = new_points, new_gt, new_active
        start = self.size
        end = start + len(points)
        self.points[start:end] = points
        self.gt[start:end] = gt_dynamic
        self.active[start:end] = True
        self.size = end
        return np.arange(start, end, dtype=np.int64)

    def point_view(self) -> np.ndarray:
        return self.points[: self.size]

    def gt_view(self) -> np.ndarray:
        return self.gt[: self.size]

    def active_view(self) -> np.ndarray:
        return self.active[: self.size]


def _metrics_from_map_counts(
    *,
    raw_dynamic: int,
    raw_static: int,
    active_dynamic: int,
    active_static: int,
    active_map_points: int,
    frontend_map_points: int,
    backend_removed_points: int,
) -> dict[str, Any]:
    """Map metrics over the source-point population and active map state."""
    true_positive = max(0, int(raw_dynamic) - int(active_dynamic))
    false_positive = max(0, int(raw_static) - int(active_static))
    false_negative = int(active_dynamic)
    true_negative = int(active_static)
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = true_positive / raw_dynamic if raw_dynamic else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = (
        true_positive / (true_positive + false_positive + false_negative)
        if true_positive + false_positive + false_negative
        else 0.0
    )
    static_keep = active_static / raw_static if raw_static else 1.0
    contamination = active_dynamic / active_map_points if active_map_points else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "static_preservation": float(static_keep),
        "map_f1": float(f1),
        "map_ghost_contamination": float(contamination),
        "map_static_completeness": float(static_keep),
        "map_dynamic_retention": float(active_dynamic / raw_dynamic) if raw_dynamic else 0.0,
        "raw_points": int(raw_dynamic + raw_static),
        "raw_dynamic_points": int(raw_dynamic),
        "raw_static_points": int(raw_static),
        "frontend_map_points": int(frontend_map_points),
        "active_map_points": int(active_map_points),
        "active_dynamic_points": int(active_dynamic),
        "active_static_points": int(active_static),
        "backend_removed_points": int(backend_removed_points),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
    }


def _map_state_metrics(
    map_buffer: _GrowingMap,
    *,
    raw_dynamic: int,
    raw_static: int,
    backend_enabled: bool,
) -> dict[str, Any]:
    active = map_buffer.active_view()
    gt = map_buffer.gt_view()
    active_dynamic = int(np.count_nonzero(active & gt))
    active_static = int(np.count_nonzero(active & ~gt))
    active_map_points = active_dynamic + active_static
    backend_removed = map_buffer.size - active_map_points if backend_enabled else 0
    return _metrics_from_map_counts(
        raw_dynamic=raw_dynamic,
        raw_static=raw_static,
        active_dynamic=active_dynamic,
        active_static=active_static,
        active_map_points=active_map_points,
        frontend_map_points=map_buffer.size,
        backend_removed_points=backend_removed,
    )


def _backend_config_from_args(args: argparse.Namespace) -> mapping_backend._BackendConfig:
    return mapping_backend._BackendConfig(
        free_fraction=float(args.backend_free_fraction),
        free_floor=int(args.backend_free_floor),
        rejudge_every=int(args.backend_rejudge_every),
        slice_budget_points=int(args.backend_slice_budget),
        max_slices_per_frame=int(args.backend_max_slices_per_frame),
        max_voxels=int(args.backend_max_voxels),
        max_recent_points=int(args.backend_max_recent_points),
        max_recent_frames=int(args.backend_max_recent_frames),
        max_queue_points=int(args.backend_max_queue_points),
        max_pending_voxels=int(args.backend_max_pending_voxels),
    )


def _gated_temporal_resolution(manifest: dict[str, Any]) -> tuple[float, float, str]:
    profile = manifest.get("sensor_profile")
    profile = profile if isinstance(profile, dict) else {}
    beams = profile.get("beams")
    if beams is not None and int(beams) <= 32:
        return 2.5, 2.5, "sensor metadata: 32-beam class"
    return 1.0, 1.0, "sensor metadata: 64-beam/default dense class"


def _replay_mapping_frontend(
    manifest: dict[str, Any],
    manifest_root: Path,
    args: argparse.Namespace,
    *,
    frontend: str,
    translation_sigma: float,
    yaw_sigma_deg: float,
    backend_enabled: bool,
) -> dict[str, Any]:
    """Replay one deterministic mapping branch, including the private backend."""
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("manifest.frames must be a non-empty list")

    temporal_h_res = temporal_v_res = None
    resolution_source = "run arguments"
    if frontend == "temporal_gated":
        temporal_h_res, temporal_v_res, resolution_source = _gated_temporal_resolution(manifest)
        filterer = _new_filter(
            args,
            algorithm="temporal",
            temporal_visibility=True,
            temporal_h_res=temporal_h_res,
            temporal_v_res=temporal_v_res,
        )
    else:
        filterer = _new_filter(args, algorithm=frontend)

    backend = (
        mapping_backend._BoundedFreeSpaceBackend(_backend_config_from_args(args))
        if backend_enabled
        else None
    )
    map_buffer = _GrowingMap()
    rng = np.random.default_rng(args.seed)
    frontend_latencies_ms: list[float] = []
    backend_frame_records: list[dict[str, Any]] = []
    map_history: list[dict[str, Any]] = []
    raw_dynamic = 0
    raw_static = 0
    fail_open_frames = 0
    dropped_frames = 0

    for index, frame in enumerate(frames):
        if not isinstance(frame, dict) or "cloud" not in frame:
            raise ValueError(f"invalid frame {index}: cloud is required")
        points = core.load_points(_resolve_path(manifest_root, frame["cloud"]), fmt="auto")
        gt_dynamic = _load_gt_mask(frame, points, manifest_root)
        raw_dynamic += int(np.count_nonzero(gt_dynamic))
        raw_static += int(np.count_nonzero(~gt_dynamic))
        pose_payload = frame.get("pose")
        fail_open = pose_payload is None
        if fail_open:
            if args.missing_pose == "error":
                raise ValueError(f"frame {index} has no pose")
            fail_open_frames += 1
            fixed_points = points
            sensor_origin = np.zeros(3, dtype=np.float64)
        else:
            pose = _perturb_pose(
                _pose_from_payload(pose_payload),
                translation_sigma=translation_sigma,
                yaw_sigma_deg=yaw_sigma_deg,
                rng=rng,
            )
            fixed_points = points @ pose.rotation.T + pose.translation
            sensor_origin = pose.translation

        started = time.perf_counter()
        if frontend == "temporal_gated":
            _, keep_mask = filterer.filter(fixed_points, sensor_origin=sensor_origin)
        elif frontend == "temporal":
            _, keep_mask = filterer.filter(fixed_points)
        else:
            _, keep_mask = filterer.filter(fixed_points, sensor_origin=sensor_origin)
        frontend_elapsed_ms = (time.perf_counter() - started) * 1000.0
        frontend_latencies_ms.append(frontend_elapsed_ms)

        accepted_points = fixed_points[keep_mask]
        accepted_gt = gt_dynamic[keep_mask]
        map_indices = map_buffer.append(accepted_points, accepted_gt)
        backend_report: dict[str, Any] = {
            "update_ms": 0.0,
            "slice_ms": 0.0,
            "slices": 0,
            "processed_points": 0,
            "removed_points": 0,
        }
        if backend is not None and not fail_open:
            backend_started = time.perf_counter()
            backend.add_map_points(accepted_points, map_indices, frame_index=index)
            backend.observe_scan(fixed_points, sensor_origin, frame_index=index)
            slice_report = backend.service(
                map_buffer.point_view(),
                map_buffer.active_view(),
                frame_index=index,
            )
            backend_report = {
                "update_ms": float((time.perf_counter() - backend_started) * 1000.0),
                "slice_ms": float(sum(slice_report["durations_ms"])),
                "slices": int(slice_report["slices"]),
                "processed_points": int(slice_report["processed_points"]),
                "removed_points": int(slice_report["removed_points"]),
            }

        frame_metrics = _map_state_metrics(
            map_buffer,
            raw_dynamic=raw_dynamic,
            raw_static=raw_static,
            backend_enabled=backend is not None,
        )
        timestamp = float(frame.get("timestamp_sec", index))
        backend_frame_records.append(
            {
                "index": index,
                "timestamp_sec": timestamp,
                "points": int(len(points)),
                "frontend_kept_points": int(len(accepted_points)),
                "gt_dynamic_points": int(np.count_nonzero(gt_dynamic)),
                "fail_open": bool(fail_open),
                "frontend_latency_ms": float(frontend_elapsed_ms),
                "backend": backend_report,
                "map_ghost_contamination": frame_metrics["map_ghost_contamination"],
                "map_static_completeness": frame_metrics["map_static_completeness"],
                "active_map_points": frame_metrics["active_map_points"],
            }
        )
        map_history.append(
            {
                "frame": index,
                "timestamp_sec": timestamp,
                "map_ghost_contamination": frame_metrics["map_ghost_contamination"],
                "map_static_completeness": frame_metrics["map_static_completeness"],
                "active_map_points": frame_metrics["active_map_points"],
            }
        )

    final_drain: dict[str, Any] | None = None
    if backend is not None:
        drain_started = time.perf_counter()
        drain = backend.drain(map_buffer.point_view(), map_buffer.active_view())
        final_drain = {
            "wall_ms": float((time.perf_counter() - drain_started) * 1000.0),
            "slices": int(drain["slices"]),
            "processed_points": int(drain["processed_points"]),
            "removed_points": int(drain["removed_points"]),
            "slice_ms": float(sum(drain["durations_ms"])),
        }

    final_metrics = _map_state_metrics(
        map_buffer,
        raw_dynamic=raw_dynamic,
        raw_static=raw_static,
        backend_enabled=backend is not None,
    )
    profile = manifest.get("sensor_profile") if isinstance(manifest.get("sensor_profile"), dict) else {}
    rate_hz = float(args.rate_hz or profile.get("rate_hz") or 0.0)
    period_ms = 1000.0 / rate_hz if rate_hz > 0.0 else None
    frontend_latency = _latency_summary(frontend_latencies_ms)
    deadline_misses = (
        int(np.count_nonzero(np.asarray(frontend_latencies_ms) > period_ms))
        if period_ms is not None
        else None
    )
    return {
        "frontend": frontend,
        "backend_enabled": bool(backend is not None),
        "pose_noise": {
            "translation_sigma_m": float(translation_sigma),
            "yaw_sigma_deg": float(yaw_sigma_deg),
            "seed": int(args.seed),
        },
        "sensor_profile": profile,
        "range_resolution": {
            "h_res_deg": float(args.range_h_res),
            "v_res_deg": float(args.range_v_res),
            "source": resolution_source,
        },
        "gated_temporal_resolution": (
            {"h_res_deg": temporal_h_res, "v_res_deg": temporal_v_res, "source": resolution_source}
            if frontend == "temporal_gated"
            else None
        ),
        "frames": len(frames),
        "frontend_latency": frontend_latency,
        "rate_hz": rate_hz or None,
        "period_ms": period_ms,
        "deadline_misses": deadline_misses,
        "dropped_frames": dropped_frames,
        "fail_open_frames": fail_open_frames,
        "metrics": final_metrics,
        "map_history": map_history,
        "per_frame": backend_frame_records,
        "backend": backend.summary() if backend is not None else None,
        "backend_final_drain": final_drain,
    }


def replay_mapping_scenario(
    manifest: dict[str, Any],
    manifest_root: Path,
    args: argparse.Namespace,
    *,
    translation_sigma: float = 0.0,
    yaw_sigma_deg: float = 0.0,
) -> dict[str, Any]:
    """Compare identical range replays with and without the private back-end."""
    primary_frontend = args.algorithm
    front_end_only = _replay_mapping_frontend(
        manifest,
        manifest_root,
        args,
        frontend=primary_frontend,
        translation_sigma=translation_sigma,
        yaw_sigma_deg=yaw_sigma_deg,
        backend_enabled=False,
    )
    front_end_plus_backend = _replay_mapping_frontend(
        manifest,
        manifest_root,
        args,
        frontend=primary_frontend,
        translation_sigma=translation_sigma,
        yaw_sigma_deg=yaw_sigma_deg,
        backend_enabled=True,
    )
    gated_temporal = _replay_mapping_frontend(
        manifest,
        manifest_root,
        args,
        frontend="temporal_gated",
        translation_sigma=translation_sigma,
        yaw_sigma_deg=yaw_sigma_deg,
        backend_enabled=False,
    )
    return {
        "name": _scenario_name(translation_sigma, yaw_sigma_deg),
        "front_end_only": front_end_only,
        "front_end_plus_backend": front_end_plus_backend,
        "gated_temporal_front_end_only": gated_temporal,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--algorithm", choices=["temporal", "range"], default="temporal")
    parser.add_argument(
        "--backend",
        choices=["none", "bounded"],
        default="none",
        help="Private Task F mapping mode; bounded compares front-end-only with +back-end.",
    )
    parser.add_argument("--voxel-size", type=float, default=core.DEFAULT_TEMPORAL_VOXEL_SIZE)
    parser.add_argument("--temporal-window", type=int, default=5)
    parser.add_argument("--temporal-min-hits", type=int, default=3)
    parser.add_argument("--temporal-visibility-h-res", type=float, default=1.0)
    parser.add_argument("--temporal-visibility-v-res", type=float, default=1.0)
    parser.add_argument("--temporal-visibility-margin", type=float, default=core.DEFAULT_RANGE_MARGIN)
    parser.add_argument("--temporal-visibility-fraction", type=float, default=0.30)
    parser.add_argument("--temporal-visibility-min-hits", type=int, default=1)
    parser.add_argument("--range-window", type=int, default=5)
    parser.add_argument("--range-margin", type=float, default=core.DEFAULT_RANGE_MARGIN)
    parser.add_argument("--range-h-res", type=float, default=core.DEFAULT_RANGE_H_RES_DEG)
    parser.add_argument("--range-v-res", type=float, default=core.DEFAULT_RANGE_V_RES_DEG)
    parser.add_argument("--backend-free-fraction", type=float, default=0.70)
    parser.add_argument("--backend-free-floor", type=int, default=3)
    parser.add_argument("--backend-rejudge-every", type=int, default=3)
    parser.add_argument("--backend-slice-budget", type=int, default=20_000)
    parser.add_argument("--backend-max-slices-per-frame", type=int, default=1)
    parser.add_argument("--backend-max-voxels", type=int, default=250_000)
    parser.add_argument("--backend-max-recent-points", type=int, default=500_000)
    parser.add_argument("--backend-max-recent-frames", type=int, default=12)
    parser.add_argument("--backend-max-queue-points", type=int, default=500_000)
    parser.add_argument("--backend-max-pending-voxels", type=int, default=250_000)
    parser.add_argument("--pose-noise-translation", type=float, nargs="*", default=[])
    parser.add_argument("--pose-noise-yaw", type=float, nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rate-hz", type=float, default=None, help="Override sensor_profile.rate_hz.")
    parser.add_argument("--missing-pose", choices=["error", "fail-open"], default="error")
    parser.add_argument("--confirmation-static-keep", type=float, default=0.95)
    parser.add_argument("--summary-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.temporal_window <= 0 or args.temporal_min_hits <= 0 or args.range_window <= 0:
        raise SystemExit("window sizes and temporal min hits must be positive")
    if not 0.0 <= args.confirmation_static_keep <= 1.0:
        raise SystemExit("--confirmation-static-keep must be between 0 and 1")
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scenarios = [(0.0, 0.0)]
    scenarios.extend((float(v), 0.0) for v in args.pose_noise_translation if float(v) > 0.0)
    scenarios.extend((0.0, float(v)) for v in args.pose_noise_yaw if float(v) > 0.0)
    if args.backend == "bounded":
        mapping_results = [
            replay_mapping_scenario(
                manifest,
                manifest_path.parent,
                args,
                translation_sigma=translation_sigma,
                yaw_sigma_deg=yaw_sigma,
            )
            for translation_sigma, yaw_sigma in scenarios
        ]
        profile = manifest.get("sensor_profile") if isinstance(manifest.get("sensor_profile"), dict) else {}
        payload = {
            "task": "online_static_mapping",
            "status": "experimental_private",
            "manifest": str(manifest_path),
            "algorithm": args.algorithm,
            "sensor_profile": profile,
            "config": {
                "range_window": args.range_window,
                "range_margin": args.range_margin,
                "range_h_res": args.range_h_res,
                "range_v_res": args.range_v_res,
                "voxel_size": args.voxel_size,
                "temporal_window": args.temporal_window,
                "temporal_min_hits": args.temporal_min_hits,
                "gated_temporal_visibility_fraction": args.temporal_visibility_fraction,
                "gated_temporal_visibility_min_hits": args.temporal_visibility_min_hits,
                "missing_pose": args.missing_pose,
                "backend": vars(_backend_config_from_args(args)),
            },
            "scenarios": mapping_results,
        }
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        for scenario in mapping_results:
            print(f"Scenario: {scenario['name']}")
            for row_name in (
                "front_end_only",
                "front_end_plus_backend",
                "gated_temporal_front_end_only",
            ):
                row = scenario[row_name]
                metrics = row["metrics"]
                print(
                    f"  {row_name}: f1={metrics['map_f1']:.3f} "
                    f"static={metrics['map_static_completeness']:.3f} "
                    f"ghost={metrics['map_ghost_contamination']:.3f} "
                    f"front_p95={row['frontend_latency']['p95_ms']:.3f}ms"
                )
        print(f"Saved: {args.summary_json}")
        return 0
    results = [
        replay_scenario(
            manifest,
            manifest_path.parent,
            args,
            translation_sigma=translation_sigma,
            yaw_sigma_deg=yaw_sigma,
        )
        for translation_sigma, yaw_sigma in scenarios
    ]
    profile = manifest.get("sensor_profile") if isinstance(manifest.get("sensor_profile"), dict) else {}
    payload = {
        "task": "online_moving_object_segmentation",
        "manifest": str(manifest_path),
        "algorithm": args.algorithm,
        "sensor_profile": profile,
        "config": {
            "voxel_size": args.voxel_size,
            "temporal_window": args.temporal_window,
            "temporal_min_hits": args.temporal_min_hits,
            "range_window": args.range_window,
            "range_margin": args.range_margin,
            "range_h_res": args.range_h_res,
            "range_v_res": args.range_v_res,
            "missing_pose": args.missing_pose,
        },
        "scenarios": results,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    baseline = results[0]
    m = baseline["metrics"]
    latency = baseline["filter_latency"]
    print(
        f"{args.algorithm}: frames={baseline['frames']} points={baseline['points']} "
        f"precision={m['precision']:.3f} recall={m['recall']:.3f} f1={m['f1']:.3f} "
        f"static={m['static_preservation']:.3f} p95={latency['p95_ms']:.3f}ms"
    )
    print(f"Saved: {args.summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
