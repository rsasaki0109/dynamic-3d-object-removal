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


# Default set of Argoverse 2 categories treated as dynamic (movable) for ground truth.
DEFAULT_DYNAMIC_LABELS = {
    "REGULAR_VEHICLE", "LARGE_VEHICLE", "BUS", "BOX_TRUCK", "TRUCK", "TRUCK_CAB",
    "VEHICULAR_TRAILER", "SCHOOL_BUS", "ARTICULATED_BUS", "MOTORCYCLE", "BICYCLE",
    "BICYCLIST", "MOTORCYCLIST", "PEDESTRIAN", "WHEELED_RIDER", "WHEELED_DEVICE",
    "WHEELCHAIR", "STROLLER", "DOG", "ANIMAL",
    # Generic / lowercase fallbacks (KITTI / JSON sources).
    "Car", "Van", "Truck", "Pedestrian", "Cyclist", "Person_sitting", "Tram", "vehicle",
}


def dynamic_gt_mask(
    points: np.ndarray,
    boxes: Sequence[core.DetectionBox],
    *,
    dynamic_labels: set[str] | None = None,
    margin: Sequence[float] = core.DEFAULT_BOX_MARGIN,
) -> np.ndarray:
    """Boolean mask of points that fall inside a *dynamic-category* annotation box.

    Reuses ``remove_points_in_boxes`` (the in-box test) and inverts the keep mask.
    """
    if points.size == 0 or len(points) == 0:
        return np.zeros(0, dtype=bool)
    if dynamic_labels is not None:
        boxes = [b for b in boxes if b.label is None or str(b.label) in dynamic_labels]
    if not boxes:
        return np.zeros(points.shape[0], dtype=bool)
    _, keep = core.remove_points_in_boxes(points, boxes, margin)
    return ~keep


def compute_accuracy_metrics(removed_mask: np.ndarray, gt_dynamic_mask: np.ndarray) -> dict[str, float]:
    """Precision/recall/F1/IoU of removed-vs-GT-dynamic, plus static preservation."""
    removed = np.asarray(removed_mask, dtype=bool)
    gt = np.asarray(gt_dynamic_mask, dtype=bool)
    tp = int(np.count_nonzero(removed & gt))
    fp = int(np.count_nonzero(removed & ~gt))
    fn = int(np.count_nonzero(~removed & gt))
    tn = int(np.count_nonzero(~removed & ~gt))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    static_preservation = tn / (tn + fp) if (tn + fp) else 1.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "static_preservation": float(static_preservation),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark dynamic object removal algorithms.")
    parser.add_argument("--input-cloud", required=True, help="Input point cloud (text/pcd/npy). For accuracy+range this is the map to clean.")
    parser.add_argument("--input-objects", help="Input detections JSON/CSV (required for box algorithm).")
    parser.add_argument(
        "--algorithm",
        choices=["box", "temporal", "range"],
        default="box",
        help="box: use detection boxes, temporal: frame consistency filter, range: range-image visibility.",
    )
    parser.add_argument(
        "--mode",
        choices=["speed", "accuracy"],
        default="speed",
        help="speed: timing/throughput (default). accuracy: precision/recall/F1/IoU vs ground-truth boxes.",
    )
    parser.add_argument("--gt-objects", help="Ground-truth boxes (e.g. AV2 annotations.feather) for accuracy mode.")
    parser.add_argument("--query-cloud", help="Query scan for accuracy+range mode (the live sweep).")
    parser.add_argument("--sensor-origin", nargs=3, type=float, default=[0.0, 0.0, 0.0], metavar=("X", "Y", "Z"), help="Sensor origin for range algorithm.")
    parser.add_argument("--timestamp-ns", type=int, default=None, help="Filter AV2 GT annotations by timestamp.")
    parser.add_argument("--iterations", type=int, default=200, help="Number of frames to benchmark.")
    parser.add_argument("--cloud-format", default="auto", choices=["auto", "csv", "pcd", "text", "npy", "feather"], help="Point cloud format.")
    parser.add_argument("--objects-format", default="auto", choices=["auto", "json", "csv", "av2", "kitti"], help="Object format.")
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


def _run_accuracy(args: argparse.Namespace) -> int:
    if not args.gt_objects:
        print("accuracy mode requires --gt-objects")
        return 1
    cloud_path = Path(args.input_cloud)
    gt_path = Path(args.gt_objects)
    if not cloud_path.exists():
        print(f"input cloud not found: {cloud_path}")
        return 1
    if not gt_path.exists():
        print(f"gt objects not found: {gt_path}")
        return 1

    points = core.load_points(cloud_path, fmt=args.cloud_format)
    gt_boxes = core.load_boxes(
        gt_path,
        fmt=args.objects_format,
        skip_invalid=True,
        timestamp_ns=args.timestamp_ns,
    )
    gt_mask = dynamic_gt_mask(points, gt_boxes, dynamic_labels=DEFAULT_DYNAMIC_LABELS)

    if args.algorithm == "range":
        query_path = Path(args.query_cloud) if args.query_cloud else None
        if not query_path or not query_path.exists():
            print("accuracy+range requires --query-cloud (the live sweep)")
            return 1
        query = core.load_points(query_path, fmt=args.cloud_format)
        _, keep = core.remove_ghost_by_range_image(points, query, tuple(args.sensor_origin))
    elif args.algorithm == "temporal":
        filt = core.TemporalConsistencyFilter(
            voxel_size=args.voxel_size,
            window_size=args.temporal_window,
            min_hits=args.temporal_min_hits,
        )
        # Prime the window with the same cloud so single-frame accuracy is well-defined.
        for _ in range(args.temporal_window):
            filt.filter(points)
        _, keep = filt.filter(points)
    else:  # box (upper bound: uses its own boxes)
        if not args.input_objects:
            print("accuracy+box requires --input-objects")
            return 1
        boxes = core.load_boxes(Path(args.input_objects), fmt=args.objects_format, skip_invalid=True, timestamp_ns=args.timestamp_ns)
        _, keep = core.remove_points_in_boxes(points, boxes, args.box_margin)

    removed_mask = ~keep
    metrics = compute_accuracy_metrics(removed_mask, gt_mask)
    summary = {
        "algorithm": args.algorithm,
        "mode": "accuracy",
        "input_points": int(points.shape[0]),
        "gt_dynamic_points": int(np.count_nonzero(gt_mask)),
        "removed_points": int(np.count_nonzero(removed_mask)),
        **metrics,
    }
    payload = {"summary": summary}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(
        "\n| algorithm | precision | recall | F1 | IoU | static kept |\n"
        "|---|---|---|---|---|---|\n"
        f"| {args.algorithm} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | "
        f"{metrics['f1']:.3f} | {metrics['iou']:.3f} | {metrics['static_preservation']:.3f} |"
    )
    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.mode == "accuracy":
        return _run_accuracy(args)

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
    elif args.algorithm == "range":
        print("algorithm=range is benchmarked with --mode accuracy (needs a map + query); see scripts/run_av2_benchmark.py")
        return 1
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
