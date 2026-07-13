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


def _new_filter(args: argparse.Namespace) -> Any:
    if args.algorithm == "temporal":
        return core.TemporalConsistencyFilter(
            voxel_size=args.voxel_size,
            window_size=args.temporal_window,
            min_hits=args.temporal_min_hits,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--algorithm", choices=["temporal", "range"], default="temporal")
    parser.add_argument("--voxel-size", type=float, default=core.DEFAULT_TEMPORAL_VOXEL_SIZE)
    parser.add_argument("--temporal-window", type=int, default=5)
    parser.add_argument("--temporal-min-hits", type=int, default=3)
    parser.add_argument("--range-window", type=int, default=5)
    parser.add_argument("--range-margin", type=float, default=core.DEFAULT_RANGE_MARGIN)
    parser.add_argument("--range-h-res", type=float, default=core.DEFAULT_RANGE_H_RES_DEG)
    parser.add_argument("--range-v-res", type=float, default=core.DEFAULT_RANGE_V_RES_DEG)
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
