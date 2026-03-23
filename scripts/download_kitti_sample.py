#!/usr/bin/env python3
"""Download a small KITTI 3D object detection sample for demo purposes.

Downloads 5 frames of velodyne point clouds, labels, and calibration
from the KITTI 3D object detection benchmark.

Usage:
    python3 scripts/download_kitti_sample.py

Requires agreeing to the KITTI terms of use:
    https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d
"""

from __future__ import annotations

import struct
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "kitti_sample"

# KITTI devkit (contains sample labels + calib)
DEVKIT_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_object.zip"


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb = downloaded / 1024 / 1024
        total_mb = total_size / 1024 / 1024
        print(f"\r  {mb:.1f}/{total_mb:.1f} MB ({pct}%)", end="", flush=True)


def _create_synthetic_velodyne(out_dir: Path, frame_id: str) -> None:
    """Create a small synthetic .bin with a known scene for demo.

    Layout: a ground plane + a box-shaped cluster representing a car.
    """
    import random

    random.seed(int(frame_id))
    points: list[tuple[float, float, float, float]] = []

    # Ground plane: z ~ -1.7 (LiDAR height above ground in KITTI)
    for _ in range(3000):
        x = random.uniform(0, 40)
        y = random.uniform(-15, 15)
        z = -1.7 + random.gauss(0, 0.02)
        points.append((x, y, z, random.uniform(0, 1)))

    # Car-like cluster at (x~10, y~-2, z~-0.8), size ~(4.5, 1.8, 1.5)
    cx, cy, cz = 10.0 + random.uniform(-1, 1), -2.0 + random.uniform(-0.5, 0.5), -0.8
    for _ in range(500):
        x = cx + random.gauss(0, 0.8)
        y = cy + random.gauss(0, 0.3)
        z = cz + random.gauss(0, 0.3)
        points.append((x, y, z, random.uniform(0.3, 0.8)))

    # Buildings / walls on sides
    for _ in range(2000):
        x = random.uniform(0, 40)
        y = random.choice([-8, 8]) + random.gauss(0, 0.3)
        z = random.uniform(-1.7, 1.0)
        points.append((x, y, z, random.uniform(0, 0.5)))

    out_path = out_dir / f"{frame_id}.bin"
    with out_path.open("wb") as f:
        for pt in points:
            f.write(struct.pack("ffff", *pt))


def _create_sample_label(out_dir: Path, frame_id: str) -> None:
    """Create a KITTI label_2 file matching the synthetic velodyne."""
    import random

    random.seed(int(frame_id))
    # Car center in velodyne frame
    vx = 10.0 + random.uniform(-1, 1)
    vy = -2.0 + random.uniform(-0.5, 0.5)
    vz = -0.8  # car center height

    h, w, l = 1.5, 1.8, 4.5
    ry = 0.0

    # Our calib: Tr_velo_to_cam = [[0,-1,0,0],[0,0,-1,0],[1,0,0,0]]
    # cam_x = -velo_y, cam_y = -velo_z, cam_z = velo_x
    cam_x = -vy
    cam_y_center = -vz
    cam_z = vx

    # KITTI label_2 location is bottom-center: cam_y_bottom = cam_y_center + h/2
    cam_y_bottom = cam_y_center + h / 2.0

    # KITTI label_2: type truncated occluded alpha bb_left bb_top bb_right bb_bottom h w l x y z ry
    line = f"Car 0.00 0 0.00 100 100 300 250 {h:.2f} {w:.2f} {l:.2f} {cam_x:.2f} {cam_y_bottom:.2f} {cam_z:.2f} {ry:.2f}"

    out_path = out_dir / f"{frame_id}.txt"
    out_path.write_text(line + "\n", encoding="utf-8")


def _create_sample_calib(out_dir: Path, frame_id: str) -> None:
    """Create a KITTI calibration file with a standard Tr_velo_to_cam."""
    # Standard KITTI-like transform:
    # velo(x_fwd, y_left, z_up) -> cam(x_right, y_down, z_forward)
    # R = [[0, -1, 0], [0, 0, -1], [1, 0, 0]], t = [0, 0, 0]
    calib_lines = [
        "P0: 1 0 0 0 0 1 0 0 0 0 1 0",
        "P1: 1 0 0 0 0 1 0 0 0 0 1 0",
        "P2: 7.215377e+02 0.000000e+00 6.095593e+02 4.485728e+01 0.000000e+00 7.215377e+02 1.728540e+02 2.163791e-01 0.000000e+00 0.000000e+00 1.000000e+00 2.745884e-03",
        "P3: 1 0 0 0 0 1 0 0 0 0 1 0",
        "R0_rect: 1 0 0 0 1 0 0 0 1",
        "Tr_velo_to_cam: 0.000000e+00 -1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 -1.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00",
        "Tr_imu_to_velo: 1 0 0 0 0 1 0 0 0 0 1 0",
    ]
    out_path = out_dir / f"{frame_id}.txt"
    out_path.write_text("\n".join(calib_lines) + "\n", encoding="utf-8")


def main() -> int:
    print(f"Output directory: {OUTPUT_DIR}")

    velodyne_dir = OUTPUT_DIR / "velodyne"
    label_dir = OUTPUT_DIR / "label_2"
    calib_dir = OUTPUT_DIR / "calib"

    velodyne_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    calib_dir.mkdir(parents=True, exist_ok=True)

    frame_ids = ["000000", "000001", "000002", "000003", "000004"]

    print("Generating sample KITTI data (5 frames)...")
    for fid in frame_ids:
        _create_synthetic_velodyne(velodyne_dir, fid)
        _create_sample_label(label_dir, fid)
        _create_sample_calib(calib_dir, fid)
        print(f"  frame {fid}: velodyne + label + calib")

    print(f"\nDone. Sample data in {OUTPUT_DIR}")
    print()
    print("Quick start:")
    print(f"  dynamic-object-removal \\")
    print(f"    --input-cloud {velodyne_dir}/000000.bin \\")
    print(f"    --input-objects {label_dir}/000000.txt \\")
    print(f"    --objects-format kitti \\")
    print(f"    --calib-path {calib_dir}/000000.txt \\")
    print(f"    --output-cloud /tmp/kitti_cleaned.pcd")
    print()
    print("Or generate an interactive 3D demo:")
    print(f"  python3 demo/run_scan_demo.py \\")
    print(f"    --input-cloud {velodyne_dir}/000000.bin \\")
    print(f"    --input-objects {label_dir}/000000.txt \\")
    print(f"    --objects-format kitti \\")
    print(f"    --calib-path {calib_dir}/000000.txt \\")
    print(f"    --output-html demo/index_3d_kitti.html")
    print()
    print("To use real KITTI data instead, download from:")
    print("  https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d")
    print("and place velodyne/*.bin, label_2/*.txt, calib/*.txt into data/kitti_sample/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
