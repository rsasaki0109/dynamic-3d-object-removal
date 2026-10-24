# Benchmarks and validation

Run commands from the repository root. [Back to README](../README.md).

For a prospective comparison of frozen neighbor candidates on three new AV2 logs, see the [selection plan, per-scene results, and reproduction guide](../examples/cli_validation/FROZEN_AV2.md).

The subsequent [32-beam nuScenes trainval check](../examples/cli_validation/FROZEN_NUSCENES.md) tests the same candidates on three new scenes from one drive and records the sparse-sensor regressions.

Its [point/scan evidence diagnosis](../examples/cli_validation/COMMON_MISS_EVIDENCE.md) traces the common misses without changing parameters.

The [nearest-return attribution](../examples/cli_validation/NEARER_RETURN_ATTRIBUTION.md) then identifies annotation tracks and official semantic classes behind the closer observations.

The [motion-label/pose audit](../examples/cli_validation/BOX_MOTION_AUDIT.md) documents the 2D nuScenes ego-pose contract and its implications for this trainval sample's 3D motion proxy.

The [GT-policy comparison CLI](../examples/cli_validation/GT_POLICY_COMPARISON.md) exports named XYZ, XY and attribute-aware label definitions and compares them against shared predictions.

The [vertical-alignment experiment](../examples/cli_validation/VERTICAL_ALIGNMENT_EXPERIMENT.md) then tests an unlabeled relative-Z estimator with spatial holdout geometry, retaining its negative result and whole-scene fallbacks.

The [AV2 laser-ID density ablation](../examples/cli_validation/LASER_DENSITY_ABLATION.md) compares complementary 64/32/16-ID subsets against dense predictions on identical rows, separating controlled thinning from point-population changes.

Its [scan-ratio evidence diagnosis](../examples/cli_validation/DENSITY_SCAN_RATIO_DIAGNOSIS.md) separates lost moving-GT votes from additional static-GT removals and closes the current validation work.

## How It Compares

[ERASOR](https://github.com/LimHyungTae/ERASOR) and [Removert](https://github.com/gisbi-kim/removert) are offline map cleaners; this project also supports online per-scan filtering. This positioning table is from their papers, not a re-run benchmark.

| | This project | ERASOR | Removert |
|---|---|---|---|
| Primary goal | Per-scan / realtime removal + map cleaning | Offline static-map cleaning | Offline static-map cleaning |
| Needs a detector / 3D boxes | `box`: yes · others: no | No | No |
| Needs poses | `box`/`temporal`: no · map cleaners: yes | Yes | Yes |
| Online / realtime | **Yes** (ROS2 node) | No (batch) | No (batch) |
| Per-scan realtime | yes | no | no |
| Offline map cleaning | yes | yes | yes |
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

### Historical single-scene proof snapshots

The upstream proof artifacts are retained here as single-scene reference points. They are useful for reproducing the checked-in images, but the multi-scene means above are the transfer-oriented headline numbers.

#### AV2 detector-free proof

Argoverse 2 — 64-beam, 12 sweeps.

| Detector-free method | Precision | Recall | F1 | Static kept |
|---|---:|---:|---:|---:|
| **`fusion`** | 0.65 | **0.66** | **0.66** | 0.97 |
| `range` | **0.68** | 0.54 | 0.60 | 0.98 |
| `scan_ratio` | 0.66 | 0.56 | 0.61 | 0.98 |
| `temporal` | 0.19 | 0.72 | 0.30 | 0.78 |

![AV2 detector-free proof](../demo/av2_gt_map_proof.png)

The AV2 proof uses 12 pose-aligned sweeps, 1,235,563 points, and 84,471 moving-track GT points. `fusion` removes 66.3% of moving GT while keeping 97.4% of static GT; boxes are used only for evaluation. [Counts and configuration](../demo/av2_gt_map_proof.json).

For short windows, use `free_votes_fraction=0.7`, `free_votes_floor=3`, and `void_min_scans=4`; long-map defaults assume 100+ scans. Sparse 32-beam data favors a coarser range-image resolution (`2.5°` on nuScenes) and should not use `fusion` by default.

#### nuScenes mini — 32-beam, 12 keyframes

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

### Compare benchmark results across changes

Save a baseline JSON before a change, rerun the same benchmark with the same
dataset selection and settings, then compare the results:

```bash
python3 scripts/compare_benchmark_results.py \
  --baseline output/before.json --candidate output/after.json \
  --report-json output/regression-report.json \
  --max-metric-drop 0.005 --max-slowdown 0.2
```

The comparator reads AV2/nuScenes single-scene or aggregate JSON, DynamicMap
results, and `run_online_benchmark.py` segmentation results. It checks every
recorded method and scene/scenario, including per-scene results behind an
aggregate mean. CLI removal summaries contain no accuracy ground truth and
are not accepted as benchmark results.

By default, any precision, recall, F1, IoU (when recorded), or static-preservation
drop fails; timing may increase by at most 20%. `--max-metric-drop 0.005`
allows a 0.5 percentage-point drop. DynamicMap's SA/DA/AA/HA percentages are
normalized to the same 0–1 scale. Online dropped/fail-open/deadline-miss counts
must not increase. Thresholds are inclusive.

Exit codes are **0** for passing, **1** for regression, and **2** for incompatible
conditions or invalid input. JSON reports contain the values, deltas, thresholds,
and individual regression flags; input errors write `comparison_error` instead
of leaving an old passing report. Missing/non-finite metrics, empty runs, changed
method sets, and mismatched evaluation metadata are errors.

AV2/nuScenes timing is the runner's whole-scene `runtime_seconds`, not per-method
latency. Online timing covers filter latency, not ROS2 serialization/publishing.
DynamicMap JSON has no timing measurements, which the report marks explicitly.
Use comparable hardware and repeat noisy timing measurements when investigating
a failure. The tool validates recorded settings, scene membership, counts, and
online manifest paths; existing result files do not prove that raw data contents
or hardware are identical. Run dataset acquisition and benchmarks separately;
the comparator does not download data or launch benchmarks.

### Validate the multi-scan CLI on an AV2 sequence

Export the exact scan poses and per-point moving GT from the existing AV2
benchmark, then validate both new CLI modes:

```bash
pip install -e ".[benchmarks]"
python scripts/run_av2_benchmark.py --frames 12 --stride 3 --fusion-workers 2 \
  --online-manifest output/cli_av2/manifest.json \
  --summary-json output/cli_av2/reference.json
python scripts/validate_multiscan_cli.py \
  --manifest output/cli_av2/manifest.json --stride 3 --workers 2 \
  --reference-summary output/cli_av2/reference.json \
  --output-dir output/cli_av2/validation
```

The validation directory must be new or empty. It contains the accumulated map,
moving-GT labels, CLI point clouds, boolean keep masks, command logs,
`baseline_api.json`, `candidate_cli.json`, and `validation.json`.
The validator requires point-for-point and mask-for-mask API/CLI equality and
checks fusion/range/scan-ratio metrics against the reference benchmark. Boxes
are used for GT only, not inference. Missing moving GT makes validation fail.

For this AV2 check, `range_scan_ratio` uses the reference benchmark's dense-sensor
range and ground settings, rather than its sparse-sensor CLI defaults. This
checks the CLI execution path; it does not establish sparse-sensor accuracy.
The existing benchmark exporter declares its clouds deskewed.

`equivalence_report.json` checks accuracy equality only. Both API and CLI JSON
also record the sum of filter-call timings, excluding CLI process startup.
Use repeated CLI validation runs in separate directories to investigate timing
regressions. A later candidate can be compared against a retained
`candidate_cli.json` using `compare_benchmark_results.py`.

To obtain row-aligned masks separately, add `--output-mask kept.npy` to either
multi-scan CLI mode. The boolean mask has one entry per input-map point;
`True` means kept.
Output cloud, mask, and summary paths must be distinct and cannot overwrite
the input map, manifest, or its scan/GT files.

A compact [real AV2 validation record](../examples/cli_validation/README.md)
is checked in separately from the downloaded data and generated point clouds.

## Downstream SLAM Proof

![AV2 downstream SLAM proof](../demo/av2_downstream_gt_map_proof.png)

The experimental [`lidarslam_ros2` integration](../examples/lidarslam_ros2/README.md) feeds exact-stamp raw and cleaned clouds to the same map backend. On AV2, both branches use 11 identical cloud/odometry pairs and byte-identical trajectories and loop-edge artifacts. Realtime `range` reduces moving-GT map points by **14.1%**, keeps **96.2%** of static-GT points, and has **21.8%** removed-point precision. This is integration evidence; the offline `fusion` result above remains the accuracy headline. [Full contract and hashes](../examples/lidarslam_ros2/av2_downstream_gt_map_proof.json).

Keep these tasks separate:

| Task | Evidence |
|---|---|
| Online moving-object segmentation | Per-scan F1/IoU, static keep, confirmation delay, latency |
| Online static mapping | Same-pose raw/cleaned ghost and structure comparison |
| Offline map cleaning | Final-map point metrics or SA/DA/AA |

`temporal` keeps its legacy hit-count behavior by default. Pass `--temporal-visibility` to enable the opt-in spherical visibility gate; ROS2 also exposes `--temporal-visibility-h-res`, `--temporal-visibility-v-res`, `--temporal-visibility-margin`, `--temporal-visibility-fraction`, and `--temporal-visibility-min-hits` (`--no-visibility` makes the default explicit). On AV2, the 3-scene mean improves from F1 0.254 to 0.586 and static points kept from 0.703 to 0.968. The vectorized filter takes about 128 ms/100k points ungated or 162 ms gated (old Counter path: 516 ms).

AV2 ground preprocessing uses an ego-frame Z cutoff before pose alignment.
New benchmark runs do not reuse that cutoff as absolute city-map Z; the range
channel records `ground_z: null` and the sensor cutoff separately. Earlier
AV2 results retain their recorded settings. See the
[coordinate-safe validation notes](../examples/cli_validation/README.md#coordinate-safe-av2-ground-preprocessing)
when comparing old and new runs.
