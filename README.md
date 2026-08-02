# Dynamic 3D Object Removal

[![Tests](https://github.com/rsasaki0109/dynamic-3d-object-removal/actions/workflows/test.yml/badge.svg)](https://github.com/rsasaki0109/dynamic-3d-object-removal/actions/workflows/test.yml)
[![Live demo](https://img.shields.io/badge/demo-live-brightgreen)](https://rsasaki0109.github.io/dynamic-3d-object-removal/demo/playground.html)
[![Release](https://img.shields.io/github/v/release/rsasaki0109/dynamic-3d-object-removal)](https://github.com/rsasaki0109/dynamic-3d-object-removal/releases)

**No GPU, numpy-only, geometry-based.** Removes dynamic objects (vehicles, pedestrians, cyclists) from LiDAR scans and accumulated maps — no deep learning, `numpy` the only dependency.

## Start Here

- **🕹️ Browser playground (no install)**: https://rsasaki0109.github.io/dynamic-3d-object-removal/demo/playground.html — the real library running client-side via Pyodide. **Box / Range / Temporal** modes, AV2 64-beam and nuScenes 32-beam presets, shareable URLs, or drop your own PCD.

  [![Browser playground demo](demo/playground_demo.gif)](https://rsasaki0109.github.io/dynamic-3d-object-removal/demo/playground.html)

- More demos: [AV2 sequence](https://rsasaki0109.github.io/dynamic-3d-object-removal/demo/index_3d_sequence_av2.html) · [single scan](https://rsasaki0109.github.io/dynamic-3d-object-removal/demo/index_3d_standalone.html) · [local sequence](https://rsasaki0109.github.io/dynamic-3d-object-removal/demo/index_3d_sequence_standalone.html)

![Before/After](demo/av2_before_after.png)

![Ghost Trail Close-up](demo/av2_zoom.png)

> 20-frame accumulated Argoverse 2 map (not a single scan): 233k ghost points (11.9% of 2M) removed, static structure preserved.

### Features

- **Five algorithms, all numpy**: `box` (per-scan crop, needs 3D boxes), `temporal` (voxel consistency, optional visibility gate), `range` (range-image visibility, Removert-style remove + revert), `scan_ratio` (ERASOR-style per-column pseudo-occupancy + ground revert), `fusion` (highest-accuracy map cleaner) — the last four are detector-free
- **Fast**: 1.5 ms for 24k points on CPU; **ROS2 realtime node** (`box` / `temporal` / `range`)
- **Minimal dependencies**: `numpy` only (`pyarrow` just for Argoverse 2 Feather input)

### Which algorithm?

Every branch is backed by a measurement below:

```mermaid
flowchart TD
    Q1{"Do you have 3D boxes<br/>(detector or annotations)?"}
    Q2{"Filtering live, per scan<br/>(e.g. ROS2 in a SLAM pipeline)?"}
    Q3{"Sensor density?"}
    BOX["<b>box</b><br/>geometric crop per scan"]
    RT["<b>temporal</b> (fastest, simplest)<br/>or <b>range</b>"]
    FUSION["<b>fusion</b> — highest accuracy<br/>(Semantic-KITTI AA 98.6 / 98.0)"]
    SPARSE["<b>range</b> sized to beam density,<br/>optionally ∧ <b>scan_ratio</b> mask"]
    Q1 -- "yes" --> BOX
    Q1 -- "no" --> Q2
    Q2 -- "yes" --> RT
    Q2 -- "no — offline map cleaning" --> Q3
    Q3 -- "dense (64-beam+)" --> FUSION
    Q3 -- "sparse (≤ 32-beam)" --> SPARSE
```

## How It Compares

[ERASOR](https://github.com/LimHyungTae/ERASOR) (RA-L '21) and [Removert](https://github.com/gisbi-kim/removert) (IROS '20) clean a *finished, pose-aligned map offline*; this project also covers *online, per-scan* use. Positioning guide (from their papers, **not** a re-run benchmark):

| | **This project** | ERASOR | Removert |
|---|---|---|---|
| Primary goal | Per-scan / realtime removal + map cleaning | Offline static-map cleaning | Offline static-map cleaning |
| Needs a detector / 3D boxes | `box`: yes · others: no | No | No |
| Needs poses | `box`/`temporal`: no · map cleaners: yes | Yes | Yes |
| Online / realtime | **Yes** (ROS2 node) | No (batch) | No (batch) |
| Core stack | `numpy` only | C++ / ROS / PCL | C++ / ROS / PCL |

### Measured on Argoverse 2 (64-beam, 12 sweeps; 3-scene mean)

Detector-free methods only, reproducible with one command, no signup. Ground truth = points on objects whose track actually moved (parked cars don't count against a motion-based method). The three logs were selected by annotation-only screening for moving content; parameters were not tuned per scene.

| method (detector-free, 3-scene mean) | precision | recall | F1 | static points kept |
|---|---|---|---|---|
| **free-space fusion** (`fusion`, short-window thresholds) | 0.571 | **0.745** | **0.642** | 0.964 |
| range-image visibility (`range`) | 0.594 | 0.636 | 0.606 | 0.972 |
| scan-ratio pseudo-occupancy (`scan_ratio`) | **0.850** | 0.473 | 0.573 | **0.994** |
| temporal consistency (`temporal`, ungated) | 0.152 | **0.817** | 0.254 | 0.703 |
| temporal consistency (`temporal`, visibility-gated) | 0.556 | 0.629 | 0.586 | 0.968 |

> Logs `0b5142c1…`, `04994d08…`, and `05fa5048…` were selected by annotation-only screening for moving content. `fusion` needs relaxed short-window thresholds here (`0.7 / 3 / 4`, the script's defaults — the library defaults assume 100+ scans and drop F1 to 0.39). `range` is tunable toward precision (`--min-see-through 4` → ≈ 0.89). `scan_ratio` reaches a similar F1 through an independent signal (column occupancy vs line-of-sight); use a small fixed `--sr-min-votes` on short windows.

```bash
pip install awscli pyarrow
python3 scripts/run_av2_benchmark.py --scenes 0b5142c1-420b-3fea-9e98-b87327ae22c6 04994d08-156c-3018-9717-ba0e29be8153 05fa5048-f355-3274-b565-c0ddc547b315
```

### Also measured on nuScenes (32-beam, sparse; 6-scene eligible mean)

On a ~5× sparser sensor the one change that matters: **match the range-image resolution to beam density** (`2.5°` vs AV2's `1.0°`).
Single-scene results overstate transfer; means over all eligible mini scenes are reported instead. The mean is unweighted over six scenes; scenes with fewer than 5,000 GT dynamic points are listed by the benchmark but excluded from the mean.

| method (detector-free, 6-scene eligible mean) | precision | recall | F1 | static points kept |
|---|---|---|---|---|
| **range ∧ scan-ratio** (intersection) | **0.297** | 0.263 | **0.240** | **0.931** |
| temporal consistency (`temporal`, visibility-gated) | 0.251 | 0.382 | 0.236 | 0.880 |
| scan-ratio pseudo-occupancy (`scan_ratio`) | 0.247 | 0.331 | 0.231 | 0.825 |
| range-image visibility (`range`) | 0.127 | **0.499** | 0.187 | 0.801 |
| temporal consistency (`temporal`, ungated) | 0.107 | **0.797** | 0.158 | 0.401 |

> Busy-scene best-case example (not typical): `scene-0757`, 12 keyframes, 303,120 map points, 48,529 GT points. Its `range ∧ scan_ratio` result is F1 0.642 / static 0.842. The fine AV2 `1.0°` resolution is a poor fit here. `scan_ratio`'s column signal is more sparsity-sensitive (high recall, weak precision) — but its false positives are nearly disjoint from `range`'s, so **intersecting the two dynamic masks** gives the best precision-side numbers at no extra cost. `fusion` is not suited to sparse sensors: beyond ~13 m the beam spacing exceeds the carving voxel and static walls get carved between beams; coarser voxels don't recover it (measured F1 < 0.3).

```bash
python3 scripts/run_nuscenes_benchmark.py --scenes all   # downloads nuScenes mini once, ~3.9 GB, no signup
```

### Measured on Semantic-KITTI (DynamicMap_Benchmark)

[KTH-RPL DynamicMap_Benchmark](https://github.com/KTH-RPL/DynamicMap_Benchmark) teaser sequences (Zenodo, no signup), the benchmark's SA / DA / AA metrics. Our methods only.

| method | seq 00 SA | seq 00 DA | seq 00 AA | seq 05 SA | seq 05 DA | seq 05 AA |
|---|---|---|---|---|---|---|
| **free-space fusion** (`fusion`) | 98.9 | **98.3** | **98.6** | 98.0 | **98.1** | **98.0** |
| **scan-ratio** pseudo-occupancy (`scan_ratio`) | 98.0 | 92.8 | 95.4 | 96.0 | 97.9 | 96.9 |
| **range-image visibility** (`range`) | **99.6** | 34.5 | 58.6 | **99.8** | 25.9 | 50.9 |
| temporal consistency (`temporal`) | 97.0 | 46.6 | 67.2 | 97.3 | 25.9 | 50.2 |

> seq 00: 141 scans / 17.4 M points; seq 05: 321 scans / 39.9 M points. `fusion` matches the leaderboard-topping DUFOMap on seq 00 (AA 98.6) and exceeds every listed method on seq 05 (98.0 vs 96.3); the learning-based, GPU-trained 4dNDF (AA ≈ 99) is outside this numpy-only class. Channel thresholds were tuned on these two sequences, like most leaderboard entries — cross-dataset transfer is what the AV2/nuScenes sections above measure.

```bash
python3 scripts/run_dynamicmap_benchmark.py --sequences 00 05   # ~385 MB per sequence
```

## Installation

```bash
pip install dynamic-object-removal
```

Pure-Python wheel, `numpy` the only dependency. Extras: `[ros2]` (ROS2 node), `[benchmarks]` (AV2/nuScenes scripts). From source: `git clone` + `pip install -e .`

## Quick Start On Public Data

Real [Argoverse 2](https://www.argoverse.org/av2.html) data in three commands, no signup:

```bash
# 1. Download an AV2 sample (1 sweep + annotations, ~1.3 MB)
pip install awscli pyarrow
python3 scripts/download_av2_sample.py

# 2. Remove dynamic objects (18 vehicles, 3 pedestrians, 1 bicycle, 1 wheelchair)
dynamic-object-removal \
  --input-cloud data/av2_sample/lidar/315969904359876000.feather \
  --input-objects data/av2_sample/annotations.feather \
  --timestamp-ns 315969904359876000 \
  --output-cloud output/av2_cleaned.pcd

# 3. Inspect before/after in 3D
python3 demo/run_scan_demo.py \
  --input-cloud data/av2_sample/lidar/315969904359876000.feather \
  --input-objects data/av2_sample/annotations.feather \
  --timestamp-ns 315969904359876000 \
  --max-render-points 50000 \
  --output-html demo/index_3d_av2.html
```

> Removes 3,406 of 95,381 points (3.6%); static road and buildings remain. KITTI is also supported: `scripts/download_kitti_sample.py`.

## CLI

```bash
# Box-driven (needs detected boxes)
dynamic-object-removal \
  --input-cloud /path/to/scan.pcd \
  --input-objects /path/to/objects.json \
  --output-cloud /path/to/output.xyz

# Detector-free map cleaning (swap range for scan_ratio as needed)
dynamic-object-removal \
  --algorithm range \
  --input-map accumulated_map.npy \
  --input-cloud query_sweep.npy \
  --sensor-origin 0 0 0 \
  --output-cloud cleaned_map.npy
```

## ROS2 Realtime Node

Subscribes to `PointCloud2`, filters, publishes:

```bash
# Box-driven with an external detector
dynamic-object-removal-realtime \
  --pointcloud-topic /velodyne_points \
  --objects-topic /detected_objects \
  --output-topic /cleaned_points \
  --algorithm box

# Detector-free temporal consistency
dynamic-object-removal-realtime \
  --pointcloud-topic /velodyne_points \
  --output-topic /cleaned_points \
  --algorithm temporal \
  --voxel-size 0.10 --temporal-window 5 --temporal-min-hits 3
```

`temporal` keeps its legacy hit-count behavior by default. Pass `--temporal-visibility` to enable the opt-in spherical visibility gate; ROS2 also exposes `--temporal-visibility-h-res`, `--temporal-visibility-v-res`, `--temporal-visibility-margin`, `--temporal-visibility-fraction`, and `--temporal-visibility-min-hits` (`--no-visibility` makes the default explicit). On AV2, the 3-scene mean improves from F1 0.254 to 0.586 and static points kept from 0.703 to 0.968. The vectorized filter takes about 128 ms/100k points ungated or 162 ms gated (old Counter path: 516 ms).

## Library API

```python
from pathlib import Path
from dynamic_object_removal import load_points, load_boxes, remove_points_in_boxes, save_points

points = load_points(Path("/path/to/scan.pcd"), fmt="auto")
boxes = load_boxes(Path("/path/to/objects.json"), fmt="auto", skip_invalid=True)
kept, keep_mask = remove_points_in_boxes(points, boxes, margin=(0.05, 0.05, 0.05))

save_points(Path("/path/to/output.xyz"), kept, fmt="auto")
```

Main public APIs:

- `load_points(path, fmt="auto")` / `load_boxes(path, fmt="auto", skip_invalid=False)` / `save_points(path, fmt="auto")`
- `remove_points_in_boxes(points, boxes, margin=(0.05, 0.05, 0.05))`
- `TemporalConsistencyFilter(voxel_size=0.10, window_size=5, min_hits=3, visibility=False)`
- `remove_ghost_by_range_image(map_points, query_points, sensor_origin, range_margin=0.5)` — single map-vs-scan visibility removal
- `clean_map_by_visibility(map_points, scans, min_see_through=2, max_surface_hits=2, ground_z=None, resolutions=None)` — multi-scan map cleaner (remove + revert)
- `remove_dynamic_by_scan_ratio(map_points, query_points, sensor_origin, scan_ratio_threshold=0.2, ground_margin=0.2)` — single map-vs-scan scan-ratio removal
- `clean_map_by_scan_ratio(map_points, scans, scan_ratio_threshold=0.2, min_votes=None, votes_fraction=0.5, votes_floor=3)` — multi-scan scan-ratio cleaner (`min_votes=None` = majority of each point's column revisits)
- `clean_map_by_fusion(map_points, scans, workers=1)` — highest-accuracy map cleaner
- `RangeImageGhostFilter(window_size=5, range_margin=0.5)` — streaming range-image filter for ROS2

### Range-image visibility removal

```python
# scans: list of (points_in_map_frame, sensor_origin) from the sweeps that built the map.
kept, keep_mask = clean_map_by_visibility(
    map_points, scans,
    range_margin=0.5, min_see_through=2, max_surface_hits=2, ground_z=-1.4,
)
```

A point is removed only when enough scans see *through* it **and** few confirm it as a real surface (the Removert-style *revert* guard). For higher precision pass `resolutions=[2.5, 4.0]` (multi-resolution consensus: AV2 precision 0.68 → 0.78). Try it in the playground's **Range mode**.

### Scan-ratio (pseudo-occupancy) removal

```python
kept, keep_mask = clean_map_by_scan_ratio(
    map_points, scans,
    scan_ratio_threshold=0.2, min_map_height=0.5, ground_margin=0.2,
)
```

ERASOR-style and independent of visibility: a polar column that is tall in the map but flat in a live sweep held a moving object; above-ground points are removed, the ground reverted by a per-column plane fit. Strongest on dense (64-beam+) LiDAR; on sparse sensors prefer `range` or raise `votes_fraction`.

### Free-space fusion (highest accuracy)

```python
kept, keep_mask = clean_map_by_fusion(map_points, scans, workers=6)
```

Three independent dynamic-evidence channels, OR-fused:

```mermaid
flowchart LR
    MAP["accumulated map<br/>+ per-scan (points, sensor origin)"]
    FS["<b>free-space carving</b><br/>ray-sampled, per-scan hit precedence<br/>dynamic when ≥ 90% of observers freed it"]
    EV["<b>eroded voids</b> (DUFOMap-style)<br/>hit inflation + 26-neighborhood erosion<br/>dynamic after ≥ 11 confirmed voids"]
    SR["<b>scan-ratio votes</b><br/>polar-column occupancy, fraction 0.7"]
    OR(("OR"))
    OUT["dynamic mask removed →<br/>cleaned static map"]
    MAP --> FS
    MAP --> EV
    MAP --> SR
    FS --> OR
    EV --> OR
    SR --> OR
    OR --> OUT
```

Fractional free-space voting nails transient traffic; absolute void counts catch slow movers and late leavers — the union scores high on both (KITTI AA **98.6 / 98.0**). Carving is the cost: minutes per hundred 64-beam scans with `workers=6`, vs seconds for `range`/`scan_ratio`.

**Sizing to your data**: defaults assume a long (100+ scan) dense-sensor sequence. For short windows (~12 scans) relax to `free_votes_fraction=0.7, free_votes_floor=3, void_min_scans=4`. On sparse (32-beam) sensors use `range` instead (measured on nuScenes above).

## Demo Regeneration

```bash
# Single scan
python3 demo/run_scan_demo.py \
  --input-cloud demo/actual_scan_20240820_cloud.pcd \
  --input-objects demo/actual_scan_20240820_objects.json \
  --max-render-points 220000 \
  --output-scene demo/demo_scene_single_scan.json \
  --output-html demo/index_3d_standalone.html

# Sequence (temporal-cleaned; pass --input-objects / --input-poses for box-driven, pose-aligned)
python3 demo/run_scan_sequence_demo.py \
  --input-glob "/path/to/graph/*/cloud.pcd" \
  --frame-count 12 --stride 1 --max-render-points 9000 --fps 4 \
  --voxel-size 0.35 --window-size 5 --min-hits 3 \
  --output-html demo/index_3d_sequence_standalone.html
```

The checked-in HTML demos are self-contained (sampled point data embedded).

## Supported Formats

- Point clouds: `PCD` (ASCII / binary), `CSV`, `TXT`, `XYZ`, `NPY`, `BIN` (KITTI), `Feather` (Argoverse 2)
- Bounding boxes: `JSON`, `CSV`, `KITTI label_2`, `Feather` (Argoverse 2)
- `PCD DATA binary_compressed` is not supported

## Related Work

- [UTS-RI/dynamic_object_detection](https://github.com/UTS-RI/dynamic_object_detection)

## Releasing (maintainers)

Releases publish to PyPI via [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) on tag push. Bump `__version__` in `dynamic_object_removal.py`, commit, then:

```bash
git tag v0.6.0
git push origin v0.6.0
```
