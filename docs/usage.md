# CLI and ROS2 guide

Run commands from the repository root. [Back to README](../README.md).

## Install

New here? Follow [Clean your first map](../examples/quickstart/README.md) for a
download-free example, then replace it with your own scans and poses. The guide
covers input coordinates, deskew requirements, keep masks and result inspection.

```bash
pip install dynamic-object-removal
```

Pure-Python wheel, with `numpy` as the only required dependency. Optional extras are `[ros2]` for the ROS2 node and `[benchmarks]` for AV2/nuScenes scripts. From source: `git clone` followed by `pip install -e .`.

```bash
# Box-driven single scan
dynamic-object-removal \
  --input-cloud scan.pcd --input-objects objects.json \
  --output-cloud cleaned.pcd
```

- **Five algorithms, all numpy**: `box` (per-scan crop, needs 3D boxes), `temporal` (voxel consistency, optional visibility gate), `range` (range-image visibility, Removert-style remove + revert), `scan_ratio` (ERASOR-style per-column pseudo-occupancy + ground revert), `fusion` (dense-sensor offline map cleaner) — the last four are detector-free
- **Fast**: 1.5 ms for 24k points on CPU; **ROS2 realtime node** (`box` / `temporal` / `range`)
- **Minimal dependencies**: `numpy` only (`pyarrow` just for Argoverse 2 Feather input)

```bash
# Detector-free map cleaning
dynamic-object-removal \
  --algorithm range --input-map map.npy --input-cloud sweep.npy \
  --sensor-origin 0 0 0 --output-cloud cleaned.npy
```

### Clean a map using multiple scans

Use `fusion` for dense, pose-aligned offline maps. Supply the accumulated map
and a JSON manifest of the scans that built it:

```bash
dynamic-object-removal \
  --algorithm fusion --input-map map.npy --input-manifest scans.json \
  --output-cloud output/cleaned.npy --summary-json output/summary.json
```

`scans.json` uses the same pose fields as the online benchmark manifest.
Each `pose` transforms a deskewed sensor-frame cloud into the coordinate frame
of `map.npy` (`p_map = R @ p_sensor + translation`). The translation is also
the sensor origin in that map frame:

```json
{
  "sensor_profile": {"name": "64-beam LiDAR", "beams": 64, "deskewed": true},
  "frames": [
    {
      "cloud": "scans/000.npy",
      "pose": {
        "translation": [0, 0, 0],
        "quaternion_xyzw": [0, 0, 0, 1]
      }
    },
    {
      "cloud": "scans/001.npy",
      "pose": {
        "translation": [1, 0, 0],
        "quaternion_xyzw": [0, 0, 0, 1]
      }
    }
  ]
}
```

Cloud paths are relative to the manifest's directory, or absolute. Each cloud's
format is inferred from its extension. A pose can instead provide a proper 3×3
`rotation` matrix. For clouds already expressed in the map frame, replace
`pose` with `"sensor_origin": [x, y, z]`; do not provide both. All scans must
be nonempty and finite, and `sensor_profile.deskewed` must explicitly be `true`.
The CLI applies rigid poses; it does not deskew scans or estimate poses.

For offline experiments with nuScenes keyframes, `--allow-undeskewed` accepts
an explicit `sensor_profile.deskewed: false`. Missing or non-boolean deskew
metadata still fails validation. The CLI warns about intra-sweep motion unless
`--quiet` is used, and the JSON summary preserves the original sensor profile
and records the opt-in. These experiments do not validate deskewed or live
moving-platform behavior.

Validate sparse 32-beam nuScenes mini data with the CLI defaults and, separately,
the benchmark's map-Z ground protection:

```bash
python scripts/validate_nuscenes_cli.py --scenes all \
  --frames 12 --stride 3 --output output/cli_nuscenes_validation
```

This downloads the official mini archive when necessary (about 4 GB streamed,
keeping only LiDAR keyframes and metadata). Each scene retains its manifest,
GT, commands, keep masks, cleaned points, and timings. The validator requires
exact API/CLI masks and points for both configurations. Its aggregate reports
unweighted means only for scenes with at least 5,000 moving-GT points, listing
excluded scenes explicitly. Use a new output directory for each run.

For the official nuScenes devkit multisweep near-sensor exclusion, add
`--min-distance 1` to the validator or nuScenes benchmark. Before pose alignment,
this discards points with both `abs(sensor_x) < 1` and `abs(sensor_y) < 1` meters.
The default `0` preserves historical input selection. The
[recorded comparison](../examples/cli_validation/README.md#sparse-sensor-error-diagnosis)
shows that this preprocessing materially changes static preservation and
filter evidence; compare runs only when their input selection matches.

Without `--preset`, defaults match `clean_map_by_fusion` and target long
sequences. Choose a preset explicitly; it is not inferred from sensor metadata
or scan count:

| Preset | Intended use | Free vote fraction / floor / minimum void scans |
|---|---|---|
| `long-map` | Dense offline maps with long sequences (100+ scans) | `0.9 / 2 / 11` (API defaults) |
| `short-window` | Dense offline maps with about 12 scans | `0.7 / 3 / 4` |

```bash
dynamic-object-removal \
  --algorithm fusion --preset short-window \
  --input-map map.npy --input-manifest scans.json \
  --output-cloud output/cleaned.npy --summary-json output/summary.json
```

Explicit `--fusion-*` options override the preset regardless of argument order.
For example, `--preset short-window --fusion-free-votes-floor 4` uses
`0.7 / 4 / 4`. Presets do not make fusion suitable for sparse 32-beam sensors.

Use `--fusion-workers 4` to process scans in parallel when memory permits; the
default is one worker. All fusion API parameters are available with a
`--fusion-` prefix and hyphens, for example `--fusion-free-voxel 0.3`.
The summary records point counts, scan count, sensor metadata, the selected
preset (`null` when omitted), all effective
parameters, library version, and filter/overall processing time. These counts
describe removal, not accuracy against ground truth. Sparse 32-beam data should
use the intersection workflow below.

### Clean a sparse map with range ∩ scan-ratio

`range_scan_ratio` runs both multi-scan cleaners on the same map and scans,
then removes only points that **both** mark dynamic. A point marked by just one
method is kept. Use the same `scans.json` manifest and deskew/pose contract as
the fusion example:

```bash
dynamic-object-removal \
  --algorithm range_scan_ratio \
  --input-map map.npy --input-manifest scans.json \
  --output-cloud output/cleaned.npy --summary-json output/summary.json
```

This mode defaults to a coarse `2.5° × 2.5°` range image, at least three
see-through votes, and at most five surface confirmations, following the sparse
nuScenes benchmark settings. Scan-ratio voting uses the API's revisit-normalized
defaults. The existing single-query `--algorithm range` keeps its `0.4°`
resolution defaults.

Tune the range channel with `--range-h-res`, `--range-v-res`,
`--range-margin`, `--range-min-see-through`, and `--range-max-surface-hits`.
`--range-resolutions 2.5 4` requires consensus at both square resolutions and
overrides the horizontal/vertical resolution settings. To protect ground,
provide `--range-ground-z` in map-frame coordinates; no global ground height
is inferred by default.

Scan-ratio exposes `--scan-ratio-rings`, `--scan-ratio-sectors`,
`--scan-ratio-max-range`, `--scan-ratio-threshold`,
`--scan-ratio-min-map-height`, and `--scan-ratio-ground-margin`.
Use `--scan-ratio-votes-fraction` and `--scan-ratio-votes-floor` for normalized
voting, or `--scan-ratio-min-votes` to select a fixed absolute threshold instead.

The summary records both channels' parameters and candidate counts in
`channel_removed_points`, plus the final intersection count. It does not
report accuracy without ground truth. This CLI uses the benchmark's mask
combination; reproducing its published numbers also requires its dataset
selection, preprocessing, and ground settings. Fusion presets and
`--fusion-*` options apply only to `--algorithm fusion`.

## Quick Start On Public Data

Real [Argoverse 2](https://www.argoverse.org/av2.html) data in three commands, with no signup:

```bash
pip install awscli pyarrow
python3 scripts/download_av2_sample.py

dynamic-object-removal \
  --input-cloud data/av2_sample/lidar/315969904359876000.feather \
  --input-objects data/av2_sample/annotations.feather \
  --timestamp-ns 315969904359876000 \
  --output-cloud output/av2_cleaned.pcd

python3 demo/run_scan_demo.py \
  --input-cloud data/av2_sample/lidar/315969904359876000.feather \
  --input-objects data/av2_sample/annotations.feather \
  --timestamp-ns 315969904359876000 \
  --max-render-points 50000 \
  --output-html demo/index_3d_av2.html
```

> This sample removes 3,406 of 95,381 points (3.6%); static road and buildings remain. KITTI is also supported via `scripts/download_kitti_sample.py`.

## ROS2 Realtime

```bash
dynamic-object-removal-realtime \
  --pointcloud-topic /velodyne_points --output-topic /cleaned_points \
  --algorithm range --fixed-frame odom --range-window 3 \
  --expected-rate-hz 10 --summary-json dor_summary.json
```

The same node supports detector-driven boxes and detector-free temporal filtering:

```bash
# Box-driven with an external detector
dynamic-object-removal-realtime \
  --pointcloud-topic /velodyne_points --objects-topic /detected_objects \
  --output-topic /cleaned_points --algorithm box

# Detector-free temporal consistency
dynamic-object-removal-realtime \
  --pointcloud-topic /velodyne_points --output-topic /cleaned_points \
  --algorithm temporal --voxel-size 0.10 \
  --temporal-window 5 --temporal-min-hits 3
```

On a moving platform, `temporal` and `range` require timestamped `fixed_frame <- cloud_frame` TF. Missing, invalid, or stale TF fails open: the scan is published unchanged and excluded from history. Input must already be deskewed; omit `--fixed-frame` only for a fixed sensor or clouds already expressed in one shared frame.

For one-pass sequence accuracy, latency, confirmation delay, fail-open, and pose-noise evaluation, use [`scripts/run_online_benchmark.py`](../scripts/run_online_benchmark.py).
