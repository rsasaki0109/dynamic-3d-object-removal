# Dynamic 3D Object Removal

[![Tests](https://github.com/rsasaki0109/dynamic-3d-object-removal/actions/workflows/test.yml/badge.svg)](https://github.com/rsasaki0109/dynamic-3d-object-removal/actions/workflows/test.yml)
[![Live demo](https://img.shields.io/badge/demo-live-brightgreen)](https://rsasaki0109.github.io/dynamic-3d-object-removal/demo/playground.html)
[![Release](https://img.shields.io/github/v/release/rsasaki0109/dynamic-3d-object-removal)](https://github.com/rsasaki0109/dynamic-3d-object-removal/releases)

Geometry-based LiDAR dynamic-object removal: **no GPU, no deep learning, `numpy` only**.

[Try the browser playground](https://rsasaki0109.github.io/dynamic-3d-object-removal/demo/playground.html) — Box / Range / Temporal modes, AV2 and nuScenes presets, or drop your own PCD.

[![Browser playground](demo/playground_demo.gif)](https://rsasaki0109.github.io/dynamic-3d-object-removal/demo/playground.html)

## Install

```bash
pip install dynamic-object-removal
```

```bash
# Box-driven single scan
dynamic-object-removal \
  --input-cloud scan.pcd --input-objects objects.json \
  --output-cloud cleaned.pcd

# Detector-free map cleaning
dynamic-object-removal \
  --algorithm range --input-map map.npy --input-cloud sweep.npy \
  --sensor-origin 0 0 0 --output-cloud cleaned.npy
```

## Algorithms

| Method | Use | Detector | Poses |
|---|---|---:|---:|
| `box` | Per-scan 3D-box crop | required | no |
| `temporal` | Streaming voxel consistency | no | moving sensors: yes |
| `range` | Range-image visibility + revert | no | yes |
| `scan_ratio` | Polar pseudo-occupancy + ground revert | no | yes |
| `fusion` | Offline free-space + void + scan-ratio fusion | no | yes |

Use `fusion` for dense offline maps, `range` (optionally intersected with `scan_ratio`) for sparse sensors, and `temporal` or `range` for realtime filtering.

## How It Compares

[ERASOR](https://github.com/LimHyungTae/ERASOR) and [Removert](https://github.com/gisbi-kim/removert) are offline map cleaners; this project also supports online per-scan filtering. This positioning table is from their papers, not a re-run benchmark.

| | This project | ERASOR | Removert |
|---|---|---|---|
| Per-scan realtime | yes | no | no |
| Offline map cleaning | yes | yes | yes |
| Detector required | only `box` | no | no |
| Core stack | `numpy` | C++ / ROS / PCL | C++ / ROS / PCL |

## Measured Results

### Argoverse 2 — 64-beam, 12 sweeps

| Detector-free method | Precision | Recall | F1 | Static kept |
|---|---:|---:|---:|---:|
| **`fusion`** | 0.65 | **0.66** | **0.66** | 0.97 |
| `range` | **0.68** | 0.54 | 0.60 | 0.98 |
| `scan_ratio` | 0.66 | 0.56 | 0.61 | 0.98 |
| `temporal` | 0.19 | 0.72 | 0.30 | 0.78 |

![AV2 detector-free proof](demo/av2_gt_map_proof.png)

The AV2 proof uses 12 pose-aligned sweeps, 1,235,563 points, and 84,471 moving-track GT points. `fusion` removes 66.3% of moving GT while keeping 97.4% of static GT; boxes are used only for evaluation. [Counts and configuration](demo/av2_gt_map_proof.json).

For short windows, use `free_votes_fraction=0.7`, `free_votes_floor=3`, and `void_min_scans=4`; long-map defaults assume 100+ scans. Sparse 32-beam data favors a coarser range-image resolution (`2.5°` on nuScenes) and should not use `fusion` by default.

### nuScenes mini — 32-beam, 12 keyframes

| Detector-free method | Precision | Recall | F1 | Static kept |
|---|---:|---:|---:|---:|
| **`range ∩ scan_ratio`** | **0.51** | 0.87 | **0.64** | **0.84** |
| `range` | 0.48 | **0.92** | 0.63 | 0.81 |
| `scan_ratio` | 0.36 | 0.90 | 0.51 | 0.69 |
| `fusion` | 0.16 | 0.32 | 0.22 | 0.68 |
| `temporal` | 0.07 | 0.22 | 0.11 | 0.47 |

### Semantic-KITTI — DynamicMap Benchmark

| Method | seq 00 SA | seq 00 DA | seq 00 AA | seq 05 SA | seq 05 DA | seq 05 AA |
|---|---:|---:|---:|---:|---:|---:|
| **`fusion`** | 98.9 | **98.3** | **98.6** | 98.0 | **98.1** | **98.0** |
| `scan_ratio` | 98.0 | 92.8 | 95.4 | 96.0 | 97.9 | 96.9 |
| `range` | **99.6** | 34.5 | 58.6 | **99.8** | 25.9 | 50.9 |
| `temporal` | 97.0 | 46.6 | 67.2 | 97.3 | 25.9 | 50.2 |

The separate [20-frame AV2 sequence](https://rsasaki0109.github.io/dynamic-3d-object-removal/demo/index_3d_sequence_av2.html) is a **box-driven annotation crop**. Its 233,460 removed points include parked objects and are not reported as moving-object GT.

Reproduce the benchmarks:

```bash
python3 scripts/run_av2_benchmark.py --frames 12 --stride 3 --sr-min-votes 2
python3 scripts/run_nuscenes_benchmark.py
python3 scripts/run_dynamicmap_benchmark.py --sequences 00 05
```

## ROS2 Realtime

```bash
dynamic-object-removal-realtime \
  --pointcloud-topic /velodyne_points --output-topic /cleaned_points \
  --algorithm range --fixed-frame odom --range-window 3 \
  --expected-rate-hz 10 --summary-json dor_summary.json
```

On a moving platform, `temporal` and `range` require timestamped `fixed_frame <- cloud_frame` TF. Missing, invalid, or stale TF fails open: the scan is published unchanged and excluded from history. Input must already be deskewed; omit `--fixed-frame` only for a fixed sensor or clouds already expressed in one shared frame.

For one-pass sequence accuracy, latency, confirmation delay, fail-open, and pose-noise evaluation, use [`scripts/run_online_benchmark.py`](scripts/run_online_benchmark.py).

## Downstream SLAM Proof

![AV2 downstream SLAM proof](demo/av2_downstream_gt_map_proof.png)

The experimental [`lidarslam_ros2` integration](examples/lidarslam_ros2/README.md) feeds exact-stamp raw and cleaned clouds to the same map backend. On AV2, both branches use 11 identical cloud/odometry pairs and byte-identical trajectories and loop-edge artifacts. Realtime `range` reduces moving-GT map points by **14.1%**, keeps **96.2%** of static-GT points, and has **21.8%** removed-point precision. This is integration evidence; the offline `fusion` result above remains the accuracy headline. [Full contract and hashes](examples/lidarslam_ros2/av2_downstream_gt_map_proof.json).

Keep these tasks separate:

| Task | Evidence |
|---|---|
| Online moving-object segmentation | Per-scan F1/IoU, static keep, confirmation delay, latency |
| Online static mapping | Same-pose raw/cleaned ghost and structure comparison |
| Offline map cleaning | Final-map point metrics or SA/DA/AA |

## Python API

```python
from pathlib import Path
from dynamic_object_removal import load_points, load_boxes, remove_points_in_boxes, save_points

points = load_points(Path("scan.pcd"), fmt="auto")
boxes = load_boxes(Path("objects.json"), fmt="auto")
cleaned, keep = remove_points_in_boxes(points, boxes)
save_points(Path("cleaned.pcd"), cleaned, fmt="auto")
```

Main APIs: `TemporalConsistencyFilter`, `RangeImageGhostFilter`, `clean_map_by_visibility`, `clean_map_by_scan_ratio`, and `clean_map_by_fusion`. See their docstrings and [`dynamic_object_removal.py`](dynamic_object_removal.py).

Supported point clouds: PCD (ASCII/binary), CSV, TXT, XYZ, NPY, KITTI BIN, and AV2 Feather. Boxes: JSON, CSV, KITTI `label_2`, and AV2 Feather. `PCD DATA binary_compressed` is not supported.

More demos: [single scan](https://rsasaki0109.github.io/dynamic-3d-object-removal/demo/index_3d_standalone.html) · [temporal sequence](https://rsasaki0109.github.io/dynamic-3d-object-removal/demo/index_3d_sequence_standalone.html).
