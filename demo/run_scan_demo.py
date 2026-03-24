#!/usr/bin/env python3
"""Build a 3D demo scene from a single external point cloud.

This utility reads one point cloud and an optional box annotation file,
creates filtered output, and rewrites the standalone HTML viewer data block
for quick visual verification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dynamic_object_removal as core


ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "index_3d_standalone.html"
DEFAULT_MAX_RENDER_POINTS = 220_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate standalone HTML scene from a single scan.")
    parser.add_argument("--input-cloud", required=True, help="Input point cloud path (pcd/xyz/txt/csv/npy).")
    parser.add_argument(
        "--input-objects",
        default="",
        help="Optional input boxes JSON/CSV path. If omitted, output equals input.",
    )
    parser.add_argument(
        "--cloud-format",
        default="auto",
        choices=["auto", "csv", "pcd", "text", "npy", "bin", "feather"],
        help="Input/output point cloud format.",
    )
    parser.add_argument(
        "--objects-format",
        default="auto",
        choices=["auto", "json", "csv", "kitti", "av2"],
        help="Box format if --input-objects is set.",
    )
    parser.add_argument(
        "--calib-path",
        default="",
        help="KITTI calibration file path (required when --objects-format=kitti).",
    )
    parser.add_argument(
        "--timestamp-ns",
        type=int,
        default=None,
        help="Filter AV2 annotations by timestamp (nanoseconds).",
    )
    parser.add_argument(
        "--box-margin",
        nargs=3,
        type=float,
        default=[0.05, 0.05, 0.05],
        metavar=("X", "Y", "Z"),
        help="Safety margin used by box filter.",
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip invalid box entries while loading instead of failing.",
    )
    parser.add_argument(
        "--max-render-points",
        type=int,
        default=DEFAULT_MAX_RENDER_POINTS,
        help="Maximum points embedded into browser scene per layer. 0 or negative disables downsampling.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for downsampling.",
    )
    parser.add_argument(
        "--output-scene",
        default=str(ROOT / "demo_scene_external.json"),
        help="Path to write scene JSON.",
    )
    parser.add_argument(
        "--output-html",
        default=str(ROOT / "index_3d_scan_standalone.html"),
        help="Path to write standalone viewer HTML.",
    )
    return parser.parse_args()


def _sample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    n = points.shape[0]
    if max_points <= 0 or n <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return points[idx]


def _to_list(points: np.ndarray) -> list[list[float]]:
    return points.tolist()


def main() -> int:
    args = parse_args()

    input_cloud = Path(args.input_cloud)
    if not input_cloud.exists():
        print(f"input cloud not found: {input_cloud}")
        return 1

    points = core.load_points(input_cloud, fmt=args.cloud_format)
    if points.size == 0:
        print("input cloud is empty")
        return 1

    boxes: list[core.DetectionBox] = []
    if args.input_objects:
        objects_path = Path(args.input_objects)
        if not objects_path.exists():
            print(f"objects file not found: {objects_path}")
            return 1
        calib = Path(args.calib_path) if args.calib_path else None
        boxes = core.load_boxes(
            objects_path,
            fmt=args.objects_format,
            skip_invalid=args.skip_invalid,
            calib_path=calib,
            timestamp_ns=args.timestamp_ns,
        )

    if boxes:
        kept, keep_mask = core.remove_points_in_boxes(points, boxes, args.box_margin)
        removed = points[~keep_mask]
        box_view = boxes[0]
    else:
        kept = points
        removed = np.zeros((0, 3), dtype=np.float64)
        box_view = None

    input_sample = _sample_points(points, args.max_render_points, args.random_seed)
    kept_sample = _sample_points(kept, args.max_render_points, args.random_seed)
    removed_sample = _sample_points(removed, args.max_render_points, args.random_seed)

    xmin, ymin, zmin = np.min(points, axis=0)
    xmax, ymax, zmax = np.max(points, axis=0)

    payload = {
        "meta": {
            "input_points": int(points.shape[0]),
            "output_points": int(kept.shape[0]),
            "removed_points": int(points.shape[0] - kept.shape[0]),
            "source": str(input_cloud),
            "objects": len(boxes),
        },
        "box": {
            "center": (box_view.center.tolist() if box_view is not None else [0.0, 0.0, 0.0]),
            "size": (box_view.size.tolist() if box_view is not None else [0.0, 0.0, 0.0]),
            "yaw": (float(box_view.yaw) if box_view is not None else 0.0),
            "label": (box_view.label if box_view is not None else "scan"),
        },
        "points": {
            "input": _to_list(input_sample),
            "kept": _to_list(kept_sample),
            "removed": _to_list(removed_sample),
        },
        "limits": {
            "xmin": float(xmin),
            "xmax": float(xmax),
            "ymin": float(ymin),
            "ymax": float(ymax),
            "zmin": float(zmin),
            "zmax": float(zmax),
            "box_half": (box_view.size / 2.0).tolist() if box_view is not None else None,
            "box_center": box_view.center.tolist() if box_view is not None else None,
        },
        "objects": [
            {
                "label": box.label,
                "center": box.center.tolist(),
                "size": box.size.tolist(),
                "yaw": float(box.yaw),
            }
            for box in boxes
        ],
    }

    output_scene = Path(args.output_scene)
    output_scene.parent.mkdir(parents=True, exist_ok=True)
    output_scene.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    template = TEMPLATE.read_text(encoding="utf-8")
    marker = "      const DEMO_DATA = "
    anchor = ";\n\n      if (!window.WebGLRenderingContext)"
    start = template.index(marker)
    end = template.index(anchor, start)
    plot_data = json.dumps(payload, ensure_ascii=False)
    html = template[:start] + marker + plot_data + template[end:]

    output_html = Path(args.output_html)
    output_html.write_text(html, encoding="utf-8")

    print(f"input_points={points.shape[0]}")
    print(f"output_points={kept.shape[0]}")
    print(f"removed_points={points.shape[0] - kept.shape[0]}")
    print(f"scene_json={output_scene}")
    print(f"standalone_html={output_html}")
    print(f"objects_used={len(boxes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
