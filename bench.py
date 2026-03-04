#!/usr/bin/env python3
"""Benchmark utility for dynamic object removal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import dynamic_object_removal as core


def _percentile(values: Sequence[float], percent: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, percent))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark dynamic object removal algorithms.")
    parser.add_argument("--input-cloud", required=True, help="Input point cloud (text/pcd/npy).")
    parser.add_argument("--input-objects", help="Input detections JSON/CSV (required for box algorithm).")
    parser.add_argument(
        "--algorithm",
        choices=["box", "temporal"],
        default="box",
        help="box: use detection boxes, temporal: frame consistency filter.",
    )
    parser.add_argument("--iterations", type=int, default=200, help="Number of frames to benchmark.")
    parser.add_argument("--cloud-format", default="auto", choices=["auto", "csv", "pcd", "text", "npy"], help="Point cloud format.")
    parser.add_argument("--objects-format", default="auto", choices=["auto", "json", "csv"], help="Object format.")
    parser.add_argument("--box-margin", nargs=3, type=float, default=list(core.DEFAULT_BOX_MARGIN), metavar=("X", "Y", "Z"), help="Box margin.")
    parser.add_argument("--min-size", type=float, default=0.01, help="Filter out tiny boxes.")
    parser.add_argument("--skip-invalid", action="store_true", help="Skip invalid object entries.")
    parser.add_argument("--voxel-size", type=float, default=core.DEFAULT_TEMPORAL_VOXEL_SIZE, help="Temporal voxel size.")
    parser.add_argument("--temporal-window", type=int, default=5, help="Temporal window size.")
    parser.add_argument("--temporal-min-hits", type=int, default=3, help="Temporal min hits.")
    parser.add_argument("--summary-json", help="Write benchmark summary JSON to this file.")
    parser.add_argument("--output-cloud", help="Save one filtered cloud sample.")
    return parser


def _run_box(points: np.ndarray, boxes: list[core.DetectionBox], margin: Sequence[float], iterations: int) -> dict[str, Any]:
    durations: list[float] = []
    filtered_last = points
    removed_last = 0

    import time

    # warm-up
    core.remove_points_in_boxes(points, boxes, margin)
    for _ in range(iterations):
        start = time.perf_counter()
        filtered, keep = core.remove_points_in_boxes(points, boxes, margin)
        durations.append((time.perf_counter() - start) * 1000.0)
        filtered_last = filtered
        removed_last = int(np.count_nonzero(~keep))

    durations_arr = np.asarray(durations, dtype=np.float64)
    removed_ratio = (removed_last / len(points)) if len(points) else 0.0
    return {
        "algorithm": "box",
        "iterations": iterations,
        "input_points": int(points.shape[0]),
        "output_points_last": int(filtered_last.shape[0]),
        "removed_points_last": removed_last,
        "removed_ratio_last": float(removed_ratio),
        "elapsed_ms_mean": float(durations_arr.mean()),
        "elapsed_ms_p50": float(np.percentile(durations_arr, 50)),
        "elapsed_ms_p95": _percentile(durations_arr, 95),
        "elapsed_ms_max": float(durations_arr.max()),
        "elapsed_ms_min": float(durations_arr.min()),
    }, filtered_last


def _run_temporal(points: np.ndarray, iterations: int, voxel_size: float, window: int, min_hits: int) -> dict[str, Any]:
    filterer = core.TemporalConsistencyFilter(
        voxel_size=voxel_size,
        window_size=window,
        min_hits=min_hits,
    )
    durations: list[float] = []
    filtered_last = points
    removed_last = 0
    import time

    # warm-up
    filterer.filter(points)
    for _ in range(iterations):
        start = time.perf_counter()
        filtered, keep = filterer.filter(points)
        durations.append((time.perf_counter() - start) * 1000.0)
        filtered_last = filtered
        removed_last = int(np.count_nonzero(~keep))

    durations_arr = np.asarray(durations, dtype=np.float64)
    removed_ratio = (removed_last / len(points)) if len(points) else 0.0
    return {
        "algorithm": "temporal",
        "iterations": iterations,
        "input_points": int(points.shape[0]),
        "output_points_last": int(filtered_last.shape[0]),
        "removed_points_last": removed_last,
        "removed_ratio_last": float(removed_ratio),
        "elapsed_ms_mean": float(durations_arr.mean()),
        "elapsed_ms_p50": float(np.percentile(durations_arr, 50)),
        "elapsed_ms_p95": _percentile(durations_arr, 95),
        "elapsed_ms_max": float(durations_arr.max()),
        "elapsed_ms_min": float(durations_arr.min()),
    }, filtered_last


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    cloud_path = Path(args.input_cloud)
    if not cloud_path.exists():
        print(f"input cloud not found: {cloud_path}")
        return 1

    points = core.load_points(cloud_path, fmt=args.cloud_format)
    iterations = max(1, int(args.iterations))
    if args.algorithm == "box":
        if not args.input_objects:
            print("algorithm=box requires --input-objects")
            return 1
        obj_path = Path(args.input_objects)
        if not obj_path.exists():
            print(f"input object file not found: {obj_path}")
            return 1
        boxes = core.load_boxes(obj_path, fmt=args.objects_format, skip_invalid=args.skip_invalid)
        boxes = [b for b in boxes if (b.size >= args.min_size).all()]
        summary, filtered = _run_box(
            points,
            boxes,
            args.box_margin,
            iterations,
        )
    else:
        summary, filtered = _run_temporal(
            points,
            iterations,
            voxel_size=args.voxel_size,
            window=args.temporal_window,
            min_hits=args.temporal_min_hits,
        )

    # throughput
    total_points = max(1, summary["input_points"])
    fps = total_points / (summary["elapsed_ms_mean"] / 1000.0) if summary["elapsed_ms_mean"] > 0 else 0.0
    summary["throughput_kpps_mean"] = fps / 1000.0

    payload = {"summary": summary}
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.summary_json:
        Path(args.summary_json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if args.output_cloud:
        suffix = Path(args.output_cloud).suffix.lower()
        out_fmt = suffix[1:] if suffix else "text"
        if out_fmt not in {"pcd", "npy", "npz", "csv", "text", "auto"}:
            out_fmt = "text"
        core.save_points(Path(args.output_cloud), filtered, fmt=out_fmt)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
