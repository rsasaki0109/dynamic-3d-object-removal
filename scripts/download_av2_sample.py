#!/usr/bin/env python3
"""Download an Argoverse 2 sample scene for demo purposes.

Downloads 1 LiDAR sweep + annotations from the Argoverse 2 Sensor Dataset.
No registration required -- data is publicly available on AWS S3.

Usage:
    python3 scripts/download_av2_sample.py

Requires: awscli (pip install awscli)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCENE = "02678d04-cc9f-3148-9f95-1ba66347dff9"
TIMESTAMP = "315969904359876000"
S3_BASE = f"s3://argoverse/datasets/av2/sensor/val/{SCENE}"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "av2_sample"


def _run(cmd: list[str]) -> None:
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}", file=sys.stderr)
        raise SystemExit(1)


def main() -> int:
    print(f"Output directory: {OUTPUT_DIR}")
    lidar_dir = OUTPUT_DIR / "lidar"
    lidar_dir.mkdir(parents=True, exist_ok=True)

    print("\nDownloading Argoverse 2 sample (1 LiDAR sweep + annotations)...")

    # Download LiDAR sweep
    lidar_file = lidar_dir / f"{TIMESTAMP}.feather"
    if not lidar_file.exists():
        _run([
            "aws", "s3", "cp", "--no-sign-request",
            f"{S3_BASE}/sensors/lidar/{TIMESTAMP}.feather",
            str(lidar_file),
        ])
    else:
        print(f"  lidar already exists: {lidar_file}")

    # Download annotations
    ann_file = OUTPUT_DIR / "annotations.feather"
    if not ann_file.exists():
        _run([
            "aws", "s3", "cp", "--no-sign-request",
            f"{S3_BASE}/annotations.feather",
            str(ann_file),
        ])
    else:
        print(f"  annotations already exists: {ann_file}")

    print(f"\nDone. Sample data in {OUTPUT_DIR}")
    print()
    print("Quick start:")
    print(f"  dynamic-object-removal \\")
    print(f"    --input-cloud {lidar_dir}/{TIMESTAMP}.feather \\")
    print(f"    --input-objects {ann_file} \\")
    print(f"    --timestamp-ns {TIMESTAMP} \\")
    print(f"    --output-cloud output/av2_cleaned.pcd")
    print()
    print("Or generate an interactive 3D demo:")
    print(f"  python3 demo/run_scan_demo.py \\")
    print(f"    --input-cloud {lidar_dir}/{TIMESTAMP}.feather \\")
    print(f"    --input-objects {ann_file} \\")
    print(f"    --timestamp-ns {TIMESTAMP} \\")
    print(f"    --max-render-points 50000 \\")
    print(f"    --output-html demo/index_3d_av2.html")
    print()
    print("Data source: Argoverse 2 Sensor Dataset (CC BY-NC-SA 4.0)")
    print("  https://www.argoverse.org/av2.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
